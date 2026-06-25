#!/usr/bin/env bash

# Self-locate the manifest from THIS script's own directory (works under a versioned folder, no
# symlink). No cd / no exit: HiveOS SOURCES this file and reads $khs / $stats from it afterwards,
# so changing the caller's cwd or calling exit would break the HiveOS agent.
__MD="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]:-$0}")")" && pwd)"
. "$__MD/h-manifest.conf"

stats_raw=`cat $CUSTOM_LOG_BASENAME.log | grep "Current hashrate is" | tail -n 1`

maxDelay=120
time_now=`date +%s`

# The miner logs with env_logger, whose default line starts "[2026-06-24T19:11:32Z INFO ...]"
# (ISO-8601 UTC, leading '['). Older builds logged "2026-06-24 19:11:32.000+02:00 [INFO ]".
# Pull the timestamp anywhere on the line (bracket/position independent) and let GNU date parse it
# (it understands both the T...Z form and the "date time+offset" form natively).
ts_field=`echo "$stats_raw" | grep -oE '[0-9]{4}-[0-9]{2}-[0-9]{2}[T ][0-9]{2}:[0-9]{2}:[0-9]{2}([.][0-9]+)?(Z|[+-][0-9]{2}:?[0-9]{2})?' | head -1`
time_rep=`date -d "$ts_field" +%s 2>/dev/null || echo 0`
diffTime=`echo $((time_now-time_rep)) | tr -d '-'`

if [ "$diffTime" -lt "$maxDelay" ]; then
        # Value is second-to-last field (before unit), unit is last field
        total_hashrate=`echo $stats_raw | awk '{print $(NF-1)}' | cut -d "." -f 1,2 --output-delimiter='' | sed 's/$/0/'`
        if [[ $stats_raw == *"Thash"* ]]; then
                total_hashrate=$(($total_hashrate*1000000000))
        elif [[ $stats_raw == *"Ghash"* ]]; then
                total_hashrate=$(($total_hashrate*1000000))
        elif [[ $stats_raw == *"Mhash"* ]]; then
                total_hashrate=$(($total_hashrate*1000))
        fi

        # GPU status
        readarray -t gpu_stats < <( jq --slurp -r -c '.[] | .busids, .brand, .temp, .fan | join(" ")' $GPU_STATS_JSON 2>/dev/null)
        busids=(${gpu_stats[0]})
        brands=(${gpu_stats[1]})
        temps=(${gpu_stats[2]})
        fans=(${gpu_stats[3]})
        gpu_count=${#busids[@]}

        hash_arr=()
        busid_arr=()
        fan_arr=()
        temp_arr=()

        if [ $(gpu-detect NVIDIA) -gt 0 ]; then
                BRAND_MINER="nvidia"
        elif [ $(gpu-detect AMD) -gt 0 ]; then
                BRAND_MINER="amd"
        fi

        for(( i=0; i < gpu_count; i++ )); do
                [[ "${brands[i]}" != $BRAND_MINER ]] && continue
                [[ "${busids[i]}" =~ ^([A-Fa-f0-9]+): ]]
                busid_arr+=($((16#${BASH_REMATCH[1]})))
                temp_arr+=(${temps[i]})
                fan_arr+=(${fans[i]})
                # Per-device line: "... [INFO ] Device #N: 5.23 Ghash/s"
                gpu_raw=`cat $CUSTOM_LOG_BASENAME.log | grep "Device #$i:" | tail -n 1`
                hashrate=`echo $gpu_raw | awk '{print $(NF-1)}' | cut -d "." -f 1,2 --output-delimiter='' | sed 's/$/0/'`
                if [[ $gpu_raw == *"Thash"* ]]; then
                        hashrate=$(($hashrate*1000000000))
                elif [[ $gpu_raw == *"Ghash"* ]]; then
                        hashrate=$(($hashrate*1000000))
                elif [[ $gpu_raw == *"Mhash"* ]]; then
                        hashrate=$(($hashrate*1000))
                fi
                hash_arr+=($hashrate)
        done

        hash_json=`printf '%s\n' "${hash_arr[@]}" | jq -cs '.'`
        bus_numbers=`printf '%s\n' "${busid_arr[@]}" | jq -cs '.'`
        fan_json=`printf '%s\n' "${fan_arr[@]}" | jq -cs '.'`
        temp_json=`printf '%s\n' "${temp_arr[@]}" | jq -cs '.'`

        uptime=$(( `date +%s` - `stat -c %Y $CUSTOM_CONFIG_FILENAME` ))

        stats=$(jq -nc \
                --argjson hs "$hash_json" \
                --arg ver "$CUSTOM_VERSION" \
                --arg ths "$total_hashrate" \
                --argjson bus_numbers "$bus_numbers" \
                --argjson fan "$fan_json" \
                --argjson temp "$temp_json" \
                --arg uptime "$uptime" \
                '{ hs: $hs, hs_units: "khs", algo: "heavyhash", ver: $ver, $uptime, $bus_numbers, $temp, $fan }')
        khs=$total_hashrate
else
        khs=0
        stats="null"
fi

echo "Log file : $CUSTOM_LOG_BASENAME.log"
echo "Time since last log entry : $diffTime"
echo "Raw stats : $stats_raw"
echo "KHS : $khs"
echo "Output : $stats"

[[ -z $khs ]] && khs=0
[[ -z $stats ]] && stats="null"
