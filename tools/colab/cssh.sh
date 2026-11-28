#!/bin/bash
# cssh.sh <session> '<cmd>': run a command on a Colab VM of the second account (HOME=~/.colab-account-home) over the SSH bridge.
S=$1; shift
exec ssh -o ProxyCommand="env HOME=$HOME/.colab-account-home colab ssh --proxy-mode --identity $HOME/.ssh/id_ed25519 -s $S" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i $HOME/.ssh/id_ed25519 -o LogLevel=ERROR root@colab-$S "$@"
