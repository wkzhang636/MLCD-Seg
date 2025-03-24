all_device=8
coco_base="./coco_base"
refcoco_json="./refcoco.jsonl"

for (( chunk_id=0; chunk_id<8; chunk_id++ ))
do
    # Define the answer path for each chunk
    # Run the Python program in the background

    CUDA_VISIBLE_DEVICES="$chunk_id" python3 test_dataset_acc.py --device-id="$chunk_id" --all-device="$all_device" --coco-base="$coco_base" --refcoco-json="$refcoco_json" &
    # Uncomment below if you need a slight delay between starting each process
    # sleep 0.1
done

# Wait for all background processes to finish
wait

merged_file="result.jsonl"
if [ -f "$merged_file" ]; then
    rm "$merged_file"
fi
# Merge all the JSONL files into one
#cat "${base_answer_path}"_*.jsonl > "${base_answer_path}.jsonl"
for ((i=0; i<8; i++)); do
  input_file="ans_file_chunk_${i}.json"
  cat "$input_file" >> "result.jsonl"
done
# remove the unmerged files
for (( chunk_id=0; chunk_id<8; chunk_id++ ))
do
    # Define the answer path for each chunk
    answer_path="ans_file_chunk_${i}.json"
    if [ -f "$answer_path" ]; then
        rm "$answer_path"
    fi
done

python3 evaluate_refcoco.py --result-dir="./"