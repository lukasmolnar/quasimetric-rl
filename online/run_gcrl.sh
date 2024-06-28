#!/bin/bash

# default args are already for the online GCRL setting

args=(
    env.kind=gcrl
    agent.actor=null
    output_folder_suffix='03'
)

exec python -m online.main "${args[@]}" "${@}"
