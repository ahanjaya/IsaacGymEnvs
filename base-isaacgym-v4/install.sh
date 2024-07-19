FILE=IsaacGym_Preview_4_Package.tar.gz
if [[ ! -f "$FILE" ]]; then
    echo "Couldn't find \`$FILE\`. Please make sure it is provided under the \`base-isaacgym-v4\` directory."
    exit 0
fi

echo "Extracting IsaacGym-v4 tar file..."
tar xzf $FILE 
echo "Installing..."
cd isaacgym/python && pip install -e .

# Torch for RTX GPUs.
pip install torch==2.0.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
