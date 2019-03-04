#!/bin/bash

# Atom file
ATOM_URL=https://github.com/atom/atom/releases/download/v1.34.0/atom-amd64.tar.gz
ATOM_FILE=atom-amd64.tar.gz

# Install directory
INSTALL_DIR="$HOME/Programas"

ATOM_FOLDER="atom-amd64"

cd $INSTALL_DIR

rm $ATOM_FILE

wget -c $ATOM_URL

rm -r $ATOM_FOLDER/*

tar -zxf $ATOM_FILE --strip=1 -C $ATOM_FOLDER
