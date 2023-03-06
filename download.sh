# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

PRESIGNED_URL=""             # replace with presigned url from email
#MODEL_SIZE="7B,13B,30B,65B"  # edit this list with the model sizes you wish to download
MODEL_SIZE="65"  # edit this list with the model sizes you wish to download
TARGET_FOLDER=""             # where all files should end up

declare -A N_SHARD_DICT

N_SHARD_DICT["7"]="0"
N_SHARD_DICT["13"]="1"
N_SHARD_DICT["30"]="3"
N_SHARD_DICT["65"]="7"

echo "Downloading tokenizer"
wget ${PRESIGNED_URL/'*'/"tokenizer.model"} -O ${TARGET_FOLDER}"/tokenizer.model"
wget ${PRESIGNED_URL/'*'/"tokenizer_checklist.chk"} -O ${TARGET_FOLDER}"/tokenizer_checklist.chk"

(cd ${TARGET_FOLDER} && md5 tokenizer_checklist.chk)

for i in ${MODEL_SIZE//,/ }
do
    echo "Downloading ${i}B"
    mkdir -p ${TARGET_FOLDER}"/${i}B"
    for s in $(seq -f "0%g" 0 ${N_SHARD_DICT[$i]})
    do
        echo "Downloading shard ${s}"
        wget ${PRESIGNED_URL/'*'/"${i}B/consolidated.${s}.pth"} -O ${TARGET_FOLDER}"/${i}B/consolidated.${s}.pth"
    done
    wget ${PRESIGNED_URL/'*'/"${i}B/params.json"} -O ${TARGET_FOLDER}"/${i}B/params.json"
    wget ${PRESIGNED_URL/'*'/"${i}B/checklist.chk"} -O ${TARGET_FOLDER}"/${i}B/checklist.chk"
    echo "Checking checksums"
    (cd ${TARGET_FOLDER}"/${i}B" && md5 checklist.chk)
done