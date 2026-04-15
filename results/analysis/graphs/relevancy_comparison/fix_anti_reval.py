#!/usr/bin/env python3
"""
Script to fix the anti-re-evaluation logic in core/main.py.
"""

# Read the file
with open('core/main.py', 'r') as f:
    lines = f.readlines()

# Find and replace the problematic line
for i, line in enumerate(lines):
    if 'output_path = Path(output_dir) if isinstance(output_dir, str) else output_dir' in line:
        # Replace the line with the correct logic
        lines[i] = '        # Ensure output_path is set correctly for anti-re-evaluation\n'
        lines.insert(i+1, '        if output_dir is None:\n')
        lines.insert(i+2, '            output_path = results_base_dir / filename\n')
        lines.insert(i+3, '        else:\n')
        lines.insert(i+4, '            output_path = Path(output_dir)\n')
        lines.insert(i+5, '            output_path.parent.mkdir(parents=True, exist_ok=True)\n')
        break

# Write the file back
with open('core/main.py', 'w') as f:
    f.writelines(lines)

print('File updated successfully')
