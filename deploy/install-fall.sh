#!/usr/bin/env bash
set -Eeuo pipefail

[[ $EUID == 0 ]] || { echo 'Run as root.' >&2; exit 2; }
release=${1:?usage: deploy/install-fall.sh RELEASE_SHA}
[[ $release =~ ^[a-f0-9]{40}$ ]] || exit 2
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
environment=/etc/mediassistrover/fall-ai.env
[[ -s $environment && $(stat -c '%u:%a' "$environment") == 0:600 ]] || {
    echo 'Provision the root-owned fall-ai.env (mode 600) before deploying.' >&2
    exit 2
}
umask 022

root=/opt/mar-fall
unit=/etc/systemd/system/fall-ai.service
exec 9>/run/lock/mar-fall-deploy.lock
flock -n 9 || { echo 'Another fall service deploy is running.' >&2; exit 1; }
getent passwd marfall >/dev/null || useradd --system --user-group --no-create-home --shell /usr/sbin/nologin marfall
install -d -m 0755 "$root/releases"
target=$root/releases/$release
previous=$(readlink "$root/current" 2>/dev/null || true)
if [[ -n $previous && ( $previous != "$root/releases/"* || ! -d $previous ) ]]; then
    echo 'Unexpected current release path.' >&2
    exit 2
fi

stage=$(mktemp -d "$root/.fall-stage.XXXXXX")
switched=false
if [[ -f $unit ]]; then cp -p "$unit" "$stage/previous.service"; fi
cleanup() {
    status=$?
    if (( status != 0 )) && $switched; then
        systemctl stop fall-ai || true
        if [[ -f $stage/previous.service ]]; then
            install -m 0644 "$stage/previous.service" "$unit"
        else
            rm -f -- "$unit"
        fi
        if [[ -n $previous ]]; then
            ln -s "$previous" "$stage/restore"
            mv -Tf "$stage/restore" "$root/current"
            systemctl daemon-reload
            systemctl restart fall-ai || true
        else
            rm -f -- "$root/current"
            systemctl daemon-reload
        fi
    fi
    rm -rf -- "$stage"
    return "$status"
}
trap cleanup EXIT

if [[ ! -d $target ]]; then
    install -d -m 0755 "$stage/release"
    install -m 0644 fall_service.py fall_detector.py yolov8n-pose.onnx requirements-fall-service.txt "$stage/release/"
    # Build the venv in its permanent location because its scripts use absolute paths.
    venv_hash=$( { sha256sum requirements-fall-service.txt; python3 -V; } | sha256sum | cut -c1-40 )
    venv=$root/venvs/$venv_hash
    if [[ ! -f $venv/.ready ]]; then
        python3 -m venv "$venv"
        "$venv/bin/python" -m pip install --only-binary=:all: --disable-pip-version-check -r requirements-fall-service.txt
        touch "$venv/.ready"
    fi
    ln -s "$venv" "$stage/release/.venv"
    mv -T "$stage/release" "$target"
fi

host=$(sed -nE 's/^FALL_BIND_HOST=([0-9.]+)$/\1/p' "$environment" | tail -1)
host=${host:-127.0.0.1}
if [[ $host == 0.0.0.0 ]]; then host=127.0.0.1; fi
ln -s "$target" "$stage/current"
switched=true
install -m 0644 deploy/fall-ai.service "$unit"
mv -Tf "$stage/current" "$root/current"
systemctl daemon-reload
systemctl restart fall-ai

for attempt in $(seq 1 20); do
    if curl -fsS --max-time 2 "http://$host:7895/health" 2>/dev/null |
       python3 -c 'import json,sys; assert json.load(sys.stdin)["state"] == "READY"' 2>/dev/null; then
        systemctl enable fall-ai >/dev/null
        echo "FALL_READY release=$release"
        exit 0
    fi
    sleep 1
done
echo 'Fall worker failed readiness. Restoring the previous release.' >&2
exit 1
