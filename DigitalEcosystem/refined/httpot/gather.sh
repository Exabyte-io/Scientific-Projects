#!/bin/bash

cp result_header.csv results.csv
for result in $( find . -name result.csv ); do
    tail -n 1 $result >> results.csv
done

cat results.csv | cut -d, -f1-6 | column -t -s,
