#!/bin/bash
set -ex

# for M in 4 16 32 256; do
for M in 4; do
	for k in {42..46}; do
		python optimized-comment-bert-ft.py $k $M
	done
done
