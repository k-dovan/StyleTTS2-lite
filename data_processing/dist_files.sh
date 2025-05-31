# Set source directory and number of split folders
src="wavs"
dest="splits"
n=10  # Number of folders

mkdir -p "$dest"
i=0

for file in "$src"/*; do
    folder_index=$((i % n))
    mkdir -p "$dest/part_$folder_index"
    mv "$file" "$dest/part_$folder_index/"
    ((i++))

    echo $((i)) moved
done

