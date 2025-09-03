#!/bin/bash
set -ex

for M in 3 15 31 63; do
	for k in {42..46}; do
		python optimized-comment-bert-ft.py $k $M
	done
done
