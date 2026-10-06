#!/bin/sh
# Runs before nginx starts (nginx image executes /docker-entrypoint.d/*.sh).
# Refuses to start without a password: the dashboard must never be public.
set -eu
if [ -z "${DASH_USER:-}" ] || [ -z "${DASH_PASS:-}" ]; then
  echo "FATAL: DASH_USER and DASH_PASS must be set — refusing to serve the dashboard without a password" >&2
  exit 1
fi
if [ "${#DASH_PASS}" -lt 12 ]; then
  echo "FATAL: DASH_PASS must be at least 12 characters" >&2
  exit 1
fi
htpasswd -bcB /etc/nginx/.htpasswd "$DASH_USER" "$DASH_PASS" >/dev/null 2>&1
chmod 640 /etc/nginx/.htpasswd
chown root:nginx /etc/nginx/.htpasswd 2>/dev/null || true
# placeholder until the first build, so the page explains itself instead of 404
mkdir -p /data/dashboard 2>/dev/null || true
echo "dashboard auth configured for user $DASH_USER"
