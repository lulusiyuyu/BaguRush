"""
clean_and_chunk.py — 清洗 + 切分 + 附元数据 + 输出 JSONL

用法:
    cd bagurush/
    python scripts/clean_and_chunk.py

输出:
    knowledge_base/chunks/all_chunks.jsonl
"""

import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ------------------------------------------------------------------ #
#  路径配置
# ------------------------------------------------------------------ #
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data_sources"
OUTPUT_DIR = PROJECT_DIR / "knowledge_base" / "chunks"
OUTPUT_FILE = OUTPUT_DIR / "all_chunks.jsonl"

# ------------------------------------------------------------------ #
#  仓库 → topic 映射
# ------------------------------------------------------------------ #
REPO_TOPIC_MAP: Dict[str, str] = {
    "AgentGuide": "llm_rag_agent",
    "CS-Notes": "cs_408",
    "core-cs-os-networks-dbms": "cs_408",
    "machine-learning-interviews": "ml_dl",
    "LLMInterviewQuestions": "llm_rag_agent",
    "LastMinuteNotes": "cs_408",
    "RAG-Interview-Questions-and-Answers-Hub": "llm_rag_agent",
    "AI-interview-cards": "ml_dl",
}

# CookBook 子目录 → topic 映射（混合仓库）
COOKBOOK_SUBDIR_MAP: Dict[str, str] = {
    "DataStructure": "cs_408",
    "数据结构和算法": "cs_408",
    "Advance高级知识": "java_backend",
    "Dubbo": "java_backend",
    "JVM": "java_backend",
    "Java8函数式编程": "java_backend",
    "Java核心": "java_backend",
    "Kafka": "java_backend",
    "MyBatis": "java_backend",
    "MySQL": "java_backend",
    "Netty": "java_backend",
    "Redis": "java_backend",
    "RocketMQ": "java_backend",
    "Spring": "java_backend",
    "SpringBoot": "java_backend",
    "SpringCloud": "java_backend",
    "Zokeeper": "java_backend",
    "分布式高并发": "java_backend",
    "架构": "java_backend",
    "设计模式": "java_backend",
    "面试汇总": "java_backend",
    "Linux": "cs_408",
    "NetWork": "cs_408",
    "Git": "java_backend",
    "Maven": "java_backend",
    "Nginx": "java_backend",
    "Python3": "cs_408",
}

# AIGC-Interview-Book 子目录 → topic 映射（混合仓库）
AIGC_SUBDIR_MAP: Dict[str, str] = {
    "AI Agent基础": "llm_rag_agent",
    "大模型基础": "llm_rag_agent",
    "大模型基础（整理中）": "llm_rag_agent",
    "AI多模态基础": "ml_dl",
    "AI绘画基础": "ml_dl",
    "AI视频基础": "ml_dl",
    "AI视频基础New": "ml_dl",
    "机器学习基础": "ml_dl",
    "深度学习基础": "ml_dl",
    "经典模型": "ml_dl",
    "模型部署基础": "ml_dl",
    "数学基础": "ml_dl",
    "数据结构基础": "cs_408",
    "计算机基础": "cs_408",
    "编程基础：C和C++": "cs_408",
    "编程基础：Python": "cs_408",
    "大厂高频算法题": "cs_408",
    "数字人基础": "ml_dl",
    "具身智能基础": "ml_dl",
    "开放性问题": "interview_exp",
    "算法岗面试求职宝典": "interview_exp",
}

# 排除的文件/目录模式
EXCLUDE_DIRS = {".git", "node_modules", "__pycache__", ".github", "assets", "imgs",
                "pictures", "images", "img", "pic", "static", "venv", ".venv"}
EXCLUDE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico",
                      ".mp4", ".mp3", ".wav", ".zip", ".tar", ".gz", ".rar",
                      ".exe", ".dll", ".so", ".dylib", ".class", ".jar",
                      ".pdf",  # PDF 单独处理
                      ".pyc", ".pyo", ".egg", ".whl",
                      ".faiss", ".pkl", ".bin", ".pt", ".pth", ".onnx",
                      ".css", ".scss", ".less",
                      ".xml", ".json", ".yaml", ".yml", ".toml",
                      ".lock", ".log"}
INCLUDE_EXTENSIONS = {".md", ".txt", ".rst", ".py", ".java", ".ipynb"}

# ------------------------------------------------------------------ #
#  辅助函数
# ------------------------------------------------------------------ #

def file_hash(content: str) -> str:
    """计算文本内容的 MD5 哈希"""
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def detect_language(text: str) -> str:
    """简单检测文本语言：中文字符占比 > 10% 则判定为中文"""
    if not text:
        return "unknown"
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    ratio = chinese_chars / max(len(text), 1)
    return "zh" if ratio > 0.1 else "en"


def infer_metadata(repo_name: str, rel_path: str, file_type: str) -> Dict:
    """
    根据仓库名和文件相对路径自动推断 metadata。
    返回 topic 和 subtopic。
    """
    parts = Path(rel_path).parts  # e.g. ("JVM", "JVM调优.md")

    # CookBook 混合仓库：按子目录映射
    if repo_name == "CookBook":
        if parts:
            top_dir = parts[0]
            topic = COOKBOOK_SUBDIR_MAP.get(top_dir, "java_backend")
        else:
            topic = "java_backend"
        subtopic = parts[0] if parts else "general"
        return {"topic": topic, "subtopic": subtopic}

    # AIGC-Interview-Book 混合仓库：按子目录映射
    if repo_name == "AIGC-Interview-Book":
        if parts:
            top_dir = parts[0]
            topic = AIGC_SUBDIR_MAP.get(top_dir, "llm_rag_agent")
        else:
            topic = "llm_rag_agent"
        subtopic = parts[0] if parts else "general"
        return {"topic": topic, "subtopic": subtopic}

    # 其他仓库：直接按 REPO_TOPIC_MAP
    topic = REPO_TOPIC_MAP.get(repo_name, "cs_408")

    # subtopic 从子目录名推断
    if len(parts) > 1:
        subtopic = parts[0]
    else:
        subtopic = Path(rel_path).stem
    return {"topic": topic, "subtopic": subtopic}


def clean_text(text: str) -> str:
    """基础文本清洗"""
    # 去除 HTML 标签（<h2>、<div>、<img>、<br> 等）
    text = re.sub(r'<[^>]+>', '', text)
    # 去除 HTML 注释
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    # 去除目录锚点行（如 - [问题](#user-content-xxx) 或 - [问题](#xxx)）
    text = re.sub(r'^\s*-\s*\[[^\]]*\]\(#[^)]*\)\s*$', '', text, flags=re.MULTILINE)
    # 去除纯 URL 行（不在 markdown 链接中的裸 URL）
    text = re.sub(r'^\s*https?://\S+\s*$', '', text, flags=re.MULTILINE)
    # 去除多余图片引用（只留 alt text）
    text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'[\1]', text)
    # 去除 HTML 实体
    text = re.sub(r'&[a-zA-Z]+;', ' ', text)
    # 去除连续空行（保留最多 2 个换行）
    text = re.sub(r'\n{4,}', '\n\n\n', text)
    # 去除行尾空白
    text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)
    return text.strip()


def read_file_safe(file_path: Path) -> Optional[str]:
    """安全读取文件，处理编码问题"""
    for encoding in ["utf-8", "gbk", "latin-1"]:
        try:
            return file_path.read_text(encoding=encoding)
        except (UnicodeDecodeError, UnicodeError):
            continue
    print(f"  [WARN] 无法读取文件（编码问题）: {file_path}")
    return None


def read_ipynb(file_path: Path) -> Optional[str]:
    """从 ipynb 中提取 markdown 和代码"""
    try:
        content = file_path.read_text(encoding="utf-8")
        notebook = json.loads(content)
        cells = notebook.get("cells", [])
        texts = []
        for cell in cells:
            cell_type = cell.get("cell_type", "")
            source = "".join(cell.get("source", []))
            if cell_type == "markdown" and source.strip():
                texts.append(source)
            elif cell_type == "code" and source.strip():
                texts.append(f"```python\n{source}\n```")
        return "\n\n".join(texts)
    except Exception as e:
        print(f"  [WARN] 无法解析 ipynb: {file_path}: {e}")
        return None


# ------------------------------------------------------------------ #
#  切分
# ------------------------------------------------------------------ #

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    递归切分文本（不依赖 langchain）。
    用于非 Markdown 文件（.txt, .py, .java 等）的通用切分。
    chunk_size 和 chunk_overlap 以字符数为单位。
    """
    separators = ["\n\n", "\n", " ", ""]

    def _split_recursive(text: str, seps: List[str]) -> List[str]:
        if len(text) <= chunk_size:
            return [text] if len(text.strip()) > 80 else []

        # 找到能用的分隔符
        sep = None
        for s in seps:
            if s and s in text:
                sep = s
                break

        # 无可用分隔符，硬切
        if sep is None:
            chunks = []
            for i in range(0, len(text), chunk_size):
                piece = text[i:i + chunk_size]
                if len(piece.strip()) > 30:
                    chunks.append(piece)
            return chunks

        parts = text.split(sep)
        chunks = []
        current = ""

        for part in parts:
            candidate = current + sep + part if current else part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                if len(part) > chunk_size:
                    idx = seps.index(sep) if sep in seps else len(seps) - 1
                    next_seps = seps[idx + 1:] if idx + 1 < len(seps) else []
                    if next_seps:
                        chunks.extend(_split_recursive(part, next_seps))
                    else:
                        for i in range(0, len(part), chunk_size):
                            piece = part[i:i + chunk_size]
                            if len(piece.strip()) > 80:
                                chunks.append(piece)
                    current = ""
                else:
                    current = part

        if current:
            chunks.append(current)

        return chunks

    raw_chunks = _split_recursive(text, separators)

    # 添加 overlap
    result = []
    for i, chunk in enumerate(raw_chunks):
        if i > 0 and chunk_overlap > 0:
            prev_tail = raw_chunks[i - 1][-chunk_overlap:]
            chunk = prev_tail + chunk
        if len(chunk.strip()) > 80:
            result.append(chunk.strip())

    return result


def chunk_by_sections(text: str, max_section_size: int = 1500, sub_chunk_size: int = 1000,
                      overlap: int = 200, min_chunk_size: int = 100) -> List[str]:
    """
    按 Markdown 标题边界切分（Structure-Aware Chunking）。

    策略：
    1. 用正则按 # / ## / ### / #### 标题拆分为独立 section
    2. 每个 section ≤ max_section_size → 整个作为 1 个 chunk（不与相邻 section 合并）
    3. 每个 section > max_section_size → 按段落二次切分为 ~sub_chunk_size 的子块
    4. 过滤掉纯链接/目录 section 和太短的 chunk（< min_chunk_size）
    5. 每个子块保留当前 section 的标题前缀

    参数：
        max_section_size: 超过此长度的 section 才进行二次切分（默认 1500 字符）
        sub_chunk_size:   二次切分的目标块大小（默认 1000 字符，对齐 BGE 模型窗口）
        overlap:          二次切分时的重叠量（默认 200 字符）
        min_chunk_size:   过滤掉短于此的 chunk（默认 100 字符）
    """
    import re

    # 第 1 步：按标题拆分成 sections
    # 匹配 1-4 级标题行（保留标题本身）
    sections = re.split(r'(?=\n#{1,4}\s)', text)

    # 处理文件开头没有标题的情况
    if sections and not sections[0].strip():
        sections = sections[1:]

    result = []

    for section in sections:
        section = section.strip()
        if not section:
            continue

        # 过滤纯链接/目录 section：去掉所有链接后不足 min_chunk_size → 跳过
        text_without_links = re.sub(r'\[([^\]]*)\]\([^)]+\)', r'\1', section)
        text_without_links = re.sub(r'https?://\S+', '', text_without_links)
        text_without_links = re.sub(r'- \[.*?\]\(#.*?\)', '', text_without_links)  # 目录锚点行
        if len(text_without_links.strip()) < min_chunk_size:
            continue

        # 第 2 步：判断是否需要二次切分
        if len(section) <= max_section_size:
            # 短 section：整个作为 1 个 chunk
            result.append(section)
        else:
            # 长 section：提取标题，按段落二次切分正文
            title_match = re.match(r'(#{1,4}\s+.*?)(?:\n|$)', section)
            title_prefix = title_match.group(1).strip() + "\n\n" if title_match else ""
            body = section[title_match.end():] if title_match else section

            # 按段落 → 行 递归切分
            sub_seps = ["\n\n", "\n", " "]
            sub_chunks = _sub_split(body, sub_seps, sub_chunk_size)

            # 添加 overlap 并保留标题前缀
            for i, chunk in enumerate(sub_chunks):
                if i > 0 and overlap > 0 and len(sub_chunks) > 1:
                    prev_tail = sub_chunks[i - 1][-overlap:]
                    chunk = prev_tail + chunk
                final = title_prefix + chunk.strip() if title_prefix else chunk.strip()
                if len(final.strip()) >= min_chunk_size:
                    result.append(final.strip())

    return result


def _sub_split(text: str, seps: List[str], target_size: int) -> List[str]:
    """辅助函数：按给定分隔符列表递归切分文本到 target_size 以下。"""
    if len(text) <= target_size:
        return [text] if text.strip() else []

    # 找到能用的分隔符
    sep = None
    for s in seps:
        if s and s in text:
            sep = s
            break

    if sep is None:
        # 硬切
        chunks = []
        for i in range(0, len(text), target_size):
            piece = text[i:i + target_size]
            if piece.strip():
                chunks.append(piece)
        return chunks

    parts = text.split(sep)
    chunks = []
    current = ""

    for part in parts:
        candidate = current + sep + part if current else part
        if len(candidate) <= target_size:
            current = candidate
        else:
            if current.strip():
                chunks.append(current)
            if len(part) > target_size:
                # 递归用下一级分隔符
                idx = seps.index(sep)
                next_seps = seps[idx + 1:]
                if next_seps:
                    chunks.extend(_sub_split(part, next_seps, target_size))
                else:
                    for i in range(0, len(part), target_size):
                        piece = part[i:i + target_size]
                        if piece.strip():
                            chunks.append(piece)
                current = ""
            else:
                current = part

    if current.strip():
        chunks.append(current)

    return chunks


# ------------------------------------------------------------------ #
#  主流程
# ------------------------------------------------------------------ #

def collect_files(data_dir: Path) -> List[Tuple[str, Path, Path]]:
    """
    遍历所有数据源，收集可处理的文件。
    返回 [(repo_name, file_path, rel_path_within_repo), ...]
    """
    files = []
    for repo_dir in sorted(data_dir.iterdir()):
        if not repo_dir.is_dir() or repo_dir.name in ("pdfs", ".git"):
            continue
        repo_name = repo_dir.name
        for root, dirs, filenames in os.walk(repo_dir):
            # 排除不需要的目录
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            for fname in filenames:
                fpath = Path(root) / fname
                suffix = fpath.suffix.lower()
                if suffix in INCLUDE_EXTENSIONS:
                    rel = fpath.relative_to(repo_dir)
                    files.append((repo_name, fpath, rel))
    return files


def process_all():
    """主处理流程"""
    print(f"[Clean&Chunk] 数据源目录: {DATA_DIR}")
    print(f"[Clean&Chunk] 输出文件: {OUTPUT_FILE}")

    if not DATA_DIR.exists():
        print("[ERROR] data_sources/ 目录不存在，请先运行 clone_sources.sh")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 收集文件
    files = collect_files(DATA_DIR)
    print(f"[Clean&Chunk] 共发现 {len(files)} 个可处理文件")

    # 2. 文件级去重（同内容 hash）
    seen_hashes = set()
    dedup_count = 0
    all_chunks = []
    chunk_id = 0

    for repo_name, fpath, rel_path in files:
        suffix = fpath.suffix.lower()

        # 读取文件
        if suffix == ".ipynb":
            raw_text = read_ipynb(fpath)
        else:
            raw_text = read_file_safe(fpath)

        if not raw_text or len(raw_text.strip()) < 50:
            continue

        # 文件级去重
        fhash = file_hash(raw_text)
        if fhash in seen_hashes:
            dedup_count += 1
            continue
        seen_hashes.add(fhash)

        # 清洗
        cleaned = clean_text(raw_text)
        if len(cleaned) < 50:
            continue

        # 推断 metadata
        file_type = suffix.lstrip(".")
        meta = infer_metadata(repo_name, str(rel_path), file_type)
        language = detect_language(cleaned)

        # 切分：Markdown 文件按标题切分，其他文件按字数切分
        if suffix in (".md", ".markdown"):
            chunks = chunk_by_sections(cleaned)
        else:
            chunks = chunk_text(cleaned)

        for chunk_text_content in chunks:
            chunk_id += 1
            chunk_obj = {
                "chunk_id": chunk_id,
                "text": chunk_text_content,
                "source_repo": repo_name,
                "source_file": rel_path.name,
                "source_path": str(rel_path),
                "topic": meta["topic"],
                "subtopic": meta["subtopic"],
                "file_type": file_type,
                "language": language,
            }
            all_chunks.append(chunk_obj)

    # 3. 写出 JSONL
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    # 4. 统计
    print(f"\n{'='*50}")
    print(f"[Clean&Chunk] 处理完成！")
    print(f"  文件总数: {len(files)}")
    print(f"  去重文件: {dedup_count}")
    print(f"  总 chunk 数: {len(all_chunks)}")
    print(f"  输出文件: {OUTPUT_FILE}")
    print(f"  文件大小: {OUTPUT_FILE.stat().st_size / 1024 / 1024:.2f} MB")

    # topic 分布
    topic_counts: Dict[str, int] = {}
    for c in all_chunks:
        t = c["topic"]
        topic_counts[t] = topic_counts.get(t, 0) + 1
    print(f"\n  Topic 分布:")
    for t, cnt in sorted(topic_counts.items(), key=lambda x: -x[1]):
        print(f"    {t}: {cnt} chunks")

    # 语言分布
    lang_counts: Dict[str, int] = {}
    for c in all_chunks:
        l = c["language"]
        lang_counts[l] = lang_counts.get(l, 0) + 1
    print(f"\n  语言分布:")
    for l, cnt in sorted(lang_counts.items(), key=lambda x: -x[1]):
        print(f"    {l}: {cnt} chunks")

    return all_chunks


if __name__ == "__main__":
    process_all()
