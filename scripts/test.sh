echo $PATH;
dir="/dccstor/graph-design/program_cache_keep-prods=2";
for file in ${dir}/*.pkl; do
    # Check if the file is a regular file (not a directory)
    if [ -f "$file" ]; then
        echo "Found file: $file"
        # Do whatever you want to do with the file here
    fi
done
