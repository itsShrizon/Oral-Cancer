import json
import os
import re

files_to_merge = [
    'configs/config.py',
    'utils/common.py',
    'utils/evaluation.py',
    'data/transforms.py',
    'data/dataset.py',
    'models/architecture.py',
    'models/loss.py',
    'engine/trainer.py',
    'train.py',
    'evaluate_final.py'
]

cells = []

# Title
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["# Dual-Head Oral Pathology Classifier"]
})

for fname in files_to_merge:
    path = os.path.join('/home/tanzir/Downloads/RepoGraveyard/Oral-Cancer', fname)
    if not os.path.exists(path):
        print(f"Skipping {fname}, file not found")
        continue

    with open(path, 'r') as f:
        code = f.read()

    # Remove multiline imports with parentheses
    code = re.sub(r'^from (configs|utils|data|models|engine)\..*?import\s*\([^)]+\)', '', code, flags=re.MULTILINE | re.DOTALL)
    
    # Remove single line imports
    code = re.sub(r'^from (configs|utils|data|models|engine)\..*import.*$', '', code, flags=re.MULTILINE)
    code = re.sub(r'^import (configs|utils|data|models|engine)\..*$', '', code, flags=re.MULTILINE)
    
    # Strip excess empty lines
    code = re.sub(r'\n{3,}', '\n\n', code)

    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [f"### `{fname}`"]
    })
    
    code_lines = [line + '\n' for line in code.split('\n')]
    if code_lines:
        code_lines[-1] = code_lines[-1].rstrip('\n')
        
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code_lines
    })

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

output_file = '/home/tanzir/Downloads/RepoGraveyard/Oral-Cancer/combined_main.ipynb'
with open(output_file, 'w') as f:
    json.dump(notebook, f, indent=1)

print(f"Created {output_file}")
