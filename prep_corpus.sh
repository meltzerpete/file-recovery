#!/usr/bin/env bash

target=corpus.txt

touch ${target}

for file in ~/raw_corpus/**/*.txt; do
    echo ${file} `cat ${file} | tr -d '\n'` >> ${target}
done

cat ${target} | wc -l
