import json
import re

def parse_directory_structure(lines, root_name="PINN4Science"):
    structure = {"modules": []}
    stack = []
    
    # 正则匹配带有 ├── 或 └── 的树节点行
    node_pattern = re.compile(r'^(?P<prefix>[│\s]*)(?P<symbol>[├└])──\s(?P<name>.+?)(?:\s+#\s*(?P<desc>.+))?$')
    
    for line in lines:
        if not line.strip():
            continue
        if line.strip().startswith(root_name):
            continue  # 跳过根目录显示

        match = node_pattern.match(line)
        if not match:
            continue

        prefix = match.group("prefix")
        name = match.group("name").strip()
        desc = match.group("desc").strip() if match.group("desc") else "No description available"

        # 缩进级别 = 每 4 个字符视作一级
        level = prefix.count('│') + prefix.count('    ')

        node = {
            "name": name,
            "path": "",
            "description": desc
        }

        if "." in name:  # 判断是文件
            node["is_code_file"] = True
        else:  # 是目录
            node["submodules"] = []
            node["files"] = []
            node["is_code_file"] = False

        while len(stack) > level:
            stack.pop()

        if not stack:  # 顶层目录
            node["path"] = f"{root_name}/{name}"
            structure["modules"].append(node)
        else:
            parent = stack[-1]
            node["path"] = f"{parent['path']}/{name}"
            if node["is_code_file"]:
                parent["files"].append(node)
            else:
                parent["submodules"].append(node)

        if not node["is_code_file"]:
            stack.append(node)

    return structure


# 使用示例
with open("tools/NEW_TREE.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

result = parse_directory_structure(lines, root_name="PINN4Science")

with open("tools/ai4pde_structure.json", "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print("✅ JSON 结构已成功生成")
