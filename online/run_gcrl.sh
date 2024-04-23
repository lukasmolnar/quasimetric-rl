#!/bin/bash

# default args are already for the online GCRL setting

args=(
    env.kind=gcrl
    agent.actor=null
)

exec python -m online.main "${args[@]}" "${@}"
