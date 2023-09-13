#!/bin/bash

wget -P faiss_index https://dl.fbaipublicfiles.com/sphere/sphere_sparse_index.tar.gz
tar -xzvf faiss_index/sphere_sparse_index.tar.gz -C faiss_index

wget -P faiss_index https://dl.fbaipublicfiles.com/sphere/sphere_dense_index.tar.gz
tar -xzvf faiss_index/sphere_dense_index.tar.gz -C faiss_index