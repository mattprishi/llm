#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

print_args=false
if [[ "${1:-}" == --print ]]; then
    print_args=true
    shift
fi
if (( $# > 2 )); then
    echo "Usage: bash serve.sh [--print] [config] [speculation]" >&2
    exit 2
fi

port="${PORT:-8000}"
if [[ ! "$port" =~ ^[1-9][0-9]{0,4}$ ]] || (( port > 65535 )); then
    echo "PORT must be a number between 1 and 65535." >&2
    exit 2
fi
args=(--host 127.0.0.1 --port "$port")
case "${1:-baseline}" in
    baseline) ;;
    tokens1024) args+=(--max-num-batched-tokens 1024) ;;
    tokens4096) args+=(--max-num-batched-tokens 4096) ;;
    tokens4096fi)
        args+=(--max-num-batched-tokens 4096 --attention-backend FLASHINFER --performance-mode interactivity)
        ;;
    seqs32) args+=(--max-num-seqs 32) ;;
    *) echo "Unknown server configuration: $1" >&2; exit 2 ;;
esac

case "${2:-off}" in
    off) ;;
    ngram|ngram8)
        draft_tokens=4
        if [[ "$2" == ngram8 ]]; then draft_tokens=8; fi
        args+=(--speculative-config
            "{\"method\":\"ngram\",\"num_speculative_tokens\":$draft_tokens,\"prompt_lookup_min\":2,\"prompt_lookup_max\":5}")
        ;;
    ngramgpu2)
        args+=(--speculative-config
            '{"method":"ngram_gpu","num_speculative_tokens":2,"prompt_lookup_min":2,"prompt_lookup_max":5}')
        ;;
    suffix)
        # Requires arctic-inference in the vLLM environment.
        # The prefix-cache endpoint does not reset a speculator's global suffix tree.
        args+=(--speculative-config
            '{"method":"suffix","num_speculative_tokens":16,"suffix_decoding_max_cached_requests":0}')
        ;;
    *) echo "Unknown speculation method: $2" >&2; exit 2 ;;
esac
if [[ -n "${MODEL_REVISION:-}" ]]; then
    args+=(--revision "$MODEL_REVISION" --tokenizer-revision "$MODEL_REVISION")
fi

if "$print_args"; then
    printf '%q ' vllm serve Qwen/Qwen3-8B "${args[@]}"
    printf '\n'
    exit
fi

export VLLM_SERVER_DEV_MODE=1
exec vllm serve Qwen/Qwen3-8B "${args[@]}"
