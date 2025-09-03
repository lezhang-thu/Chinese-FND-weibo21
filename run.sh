#!/bin/bash

for k in {42..46}; do
	python optimized-bert-ft.py $k
done
