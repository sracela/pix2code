#!/bin/bash
for file in ../code/*
do
        echo $file
        ./android-compiler.py  $file
done