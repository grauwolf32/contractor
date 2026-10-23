#!/usr/bin/env bash
# Sole owner of contractor-llm, hc-llm0 and CONTRACTOR_LLM_* iptables chains.
set -euo pipefail
test "$(id -u)" = 0
ns=contractor-llm
base=/etc/contractor-hybrid
ownership=/var/lib/contractor-hybrid/egress-owned
read -r address mtu mesh_ip mesh_net < <(python3 - <<'PY'
import json
d=json.load(open('/etc/contractor-hybrid/egress.json'))
print(d['address'],d['mtu'],d['mesh_address'],d['mesh_network'])
PY
)
ipt() { ip netns exec "$ns" iptables "$@"; }
host_rule() { local table=$1; shift; iptables -t "$table" -C "$@" 2>/dev/null || iptables -t "$table" -A "$@"; }
check_route() {
  ip netns exec "$ns" curl -4 --fail --silent --show-error \
    --connect-timeout 5 --max-time 20 --output /dev/null \
    --write-out 'OpenRouter HTTP %{http_code}\n' https://openrouter.ai/api/v1/models
}
case "$1" in
  install)
    export DEBIAN_FRONTEND=noninteractive
    apt-get -o Acquire::ForceIPv4=true -o Acquire::Retries=2 install -y -qq wireguard-tools iptables curl
    systemctl daemon-reload
    if systemctl is-active --quiet contractor-wg-egress; then
      "$0" up
    else
      systemctl enable --now contractor-wg-egress
    fi
    check_route
    ;;
  up)
    if ip netns list | awk '{print $1}' | grep -qx "$ns"; then
      test -f "$ownership" || { printf '%s\n' 'Refusing an unmanaged egress namespace' >&2; exit 1; }
    else
      test ! -e /sys/class/net/hc-llm0
      test ! -e /sys/class/net/wg-egress
      install -d -m 700 /var/lib/contractor-hybrid
      install -m 600 /dev/null "$ownership"
    fi
    if ! ip netns list | awk '{print $1}' | grep -qx "$ns"; then
      ip netns add "$ns"
      ip link add hc-llm0 type veth peer name llm0-ns
      ip link set llm0-ns netns "$ns"
      ip -n "$ns" link set llm0-ns name eth0
      ip address add 172.30.91.1/30 dev hc-llm0
      ip link set hc-llm0 up
      ip -n "$ns" address add 172.30.91.2/30 dev eth0
      ip -n "$ns" link set eth0 up
      ip -n "$ns" link set lo up
      # The UDP socket remains in the host namespace and uses its original route.
      ip link add wg-egress type wireguard
      ip link set wg-egress netns "$ns"
    fi
    ip netns exec "$ns" wg setconf wg-egress "$base/wg-egress.conf"
    ip -n "$ns" address replace "$address" dev wg-egress
    ip -n "$ns" link set wg-egress mtu "$mtu" up
    ip -n "$ns" route replace "$mesh_net" via 172.30.91.1 dev eth0
    ip -n "$ns" route replace default dev wg-egress
    mkdir -p /etc/netns/contractor-llm
    python3 - <<'PY'
import json,pathlib
d=json.load(open('/etc/contractor-hybrid/egress.json'))
pathlib.Path('/etc/netns/contractor-llm/resolv.conf').write_text(''.join('nameserver '+x+'\n' for x in d['dns']))
PY
    # Fail closed: only the tunnel, internal DB and established ingress replies.
    ipt -P INPUT DROP
    ipt -P OUTPUT DROP
    ipt -P FORWARD DROP
    ipt -F INPUT
    ipt -F OUTPUT
    ipt -A INPUT -i lo -j ACCEPT
    ipt -A OUTPUT -o lo -j ACCEPT
    ipt -A INPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
    ipt -A OUTPUT -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
    ipt -A OUTPUT -o wg-egress -j ACCEPT
    ipt -A INPUT -i eth0 -s "$mesh_net" -p tcp --dport 4000 -j ACCEPT
    ipt -A INPUT -i eth0 -s 172.30.91.1/32 -p tcp --dport 4000 -j ACCEPT
    ipt -A OUTPUT -o eth0 -d 172.30.91.1/32 -p tcp --dport 54329 -j ACCEPT
    # No IPv6 bypass when the provider profile only has an IPv4 source address.
    ip netns exec "$ns" sysctl -q -w net.ipv6.conf.all.disable_ipv6=1
    iptables -t nat -N CONTRACTOR_LLM_NAT 2>/dev/null || true
    iptables -t nat -F CONTRACTOR_LLM_NAT
    iptables -t nat -A CONTRACTOR_LLM_NAT -d "$mesh_ip/32" -p tcp --dport 4000 -j DNAT --to-destination 172.30.91.2:4000
    host_rule nat PREROUTING -j CONTRACTOR_LLM_NAT
    host_rule nat OUTPUT -j CONTRACTOR_LLM_NAT
    iptables -N CONTRACTOR_LLM_FWD 2>/dev/null || true
    iptables -F CONTRACTOR_LLM_FWD
    iptables -A CONTRACTOR_LLM_FWD -i wg-mesh -o hc-llm0 -d 172.30.91.2/32 -p tcp --dport 4000 -j ACCEPT
    iptables -A CONTRACTOR_LLM_FWD -i hc-llm0 -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
    iptables -A CONTRACTOR_LLM_FWD -j DROP
    host_rule filter FORWARD -i hc-llm0 -j CONTRACTOR_LLM_FWD
    host_rule filter FORWARD -o hc-llm0 -j CONTRACTOR_LLM_FWD
    # Required for wg-mesh -> namespace ingress; other traffic is unaffected.
    printf 'net.ipv4.ip_forward=1\n' > /etc/sysctl.d/90-contractor-hybrid.conf
    sysctl -q -w net.ipv4.ip_forward=1
    ;;
  check)
    systemctl is-active --quiet contractor-wg-egress
    ip netns exec "$ns" wg show wg-egress latest-handshakes
    check_route
    ;;
  check-kill-switch)
    check_route
    ip -n "$ns" link set wg-egress down
    restore_tunnel() {
      ip -n "$ns" link set wg-egress up
      # Linux removes interface routes when the link is taken down.
      ip -n "$ns" route replace default dev wg-egress
    }
    trap restore_tunnel EXIT
    if ip netns exec "$ns" curl -4 --silent --connect-timeout 2 --max-time 4 \
        --output /dev/null https://openrouter.ai/api/v1/models; then
      printf '%s\n' 'FAIL: traffic escaped while WireGuard was down' >&2
      exit 1
    fi
    printf '%s\n' 'PASS: no direct egress with WireGuard down'
    restore_tunnel
    trap - EXIT
    check_route
    ;;
  down)
    test -f "$ownership" || { printf '%s\n' 'Refusing to remove unmanaged resources' >&2; exit 1; }
    # Stop gateway containers before stopping this unit.
    for hook in PREROUTING OUTPUT; do iptables -t nat -D "$hook" -j CONTRACTOR_LLM_NAT 2>/dev/null || true; done
    iptables -D FORWARD -i hc-llm0 -j CONTRACTOR_LLM_FWD 2>/dev/null || true
    iptables -D FORWARD -o hc-llm0 -j CONTRACTOR_LLM_FWD 2>/dev/null || true
    iptables -t nat -F CONTRACTOR_LLM_NAT 2>/dev/null || true
    iptables -t nat -X CONTRACTOR_LLM_NAT 2>/dev/null || true
    iptables -F CONTRACTOR_LLM_FWD 2>/dev/null || true
    iptables -X CONTRACTOR_LLM_FWD 2>/dev/null || true
    ip netns delete "$ns" 2>/dev/null || true
    ip link delete hc-llm0 2>/dev/null || true
    ;;
  *) printf '%s\n' 'usage: egress.sh install|up|down|check|check-kill-switch' >&2; exit 2 ;;
esac
