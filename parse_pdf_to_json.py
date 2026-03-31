from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pypdf import PdfReader


TYPE_META = {
    "single": {"category": "单选题"},
    "multiple": {"category": "多选题"},
    "judge": {"category": "判断题"},
    "case": {"category": "案例题"},
}

SECTION_HINTS = [
    ("single", ["一、单选题", "单选题"]),
    ("multiple", ["二、多选题", "多选题"]),
    ("judge", ["三、判断题", "判断题"]),
    ("case", ["四、案例题", "案例题", "案例分析题"]),
]

FOOTER_PATTERNS = [
    re.compile(r"广东省建筑施工企业安全生产管理人员服务中心"),
]

QUESTION_START_RE = re.compile(r"^(\d{1,4})[\.．、]\s*(.+)$")
ANSWER_RE = re.compile(r"正确答案[：:]\s*([^\n\r]+)")
ANALYSIS_RE = re.compile(r"(?:解析|答案解析)[：:]\s*([^\n\r]+)")
OPTION_START_RE = re.compile(r"^([A-Z])[\.．、]\s*(.*)$")


@dataclass
class RawQuestion:
    no: str
    qtype: str
    source_page: int
    lines: List[str]


def norm_space(text: str) -> str:
    text = text.replace("\u3000", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def detect_section(line: str) -> Optional[str]:
    s = line.replace(" ", "")
    for qtype, hints in SECTION_HINTS:
        for h in hints:
            if h in s:
                return qtype
    return None


def is_footer(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    for p in FOOTER_PATTERNS:
        if p.search(s):
            return True
    return False


def build_raw_questions(pdf_path: Path) -> List[RawQuestion]:
    reader = PdfReader(str(pdf_path))
    current_type: Optional[str] = None
    current: Optional[RawQuestion] = None
    out: List[RawQuestion] = []
    last_no: Optional[int] = None

    for page_idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        raw_lines = text.replace("\r", "\n").split("\n")

        for raw_line in raw_lines:
            line = raw_line.strip()
            if not line:
                continue
            if is_footer(line):
                continue

            sec = detect_section(line)
            if sec:
                current_type = sec
                continue

            m = QUESTION_START_RE.match(line)
            if m and current_type:
                cand_no = int(m.group(1))
                # Guard against false starts such as decimal-like wrapped option text (e.g. "1.5m"),
                # while allowing stems that start with digits (e.g. "2303.5.5kW ...").
                if last_no is not None and cand_no <= last_no:
                    if current is not None:
                        current.lines.append(line)
                    continue
                if current is not None:
                    out.append(current)
                q_no = m.group(1)
                q_head = m.group(2).strip()
                current = RawQuestion(no=q_no, qtype=current_type, source_page=page_idx, lines=[q_head])
                last_no = cand_no
                continue

            if current is not None:
                current.lines.append(line)

    if current is not None:
        out.append(current)

    return out


def parse_answer_tokens(ans_raw: str, qtype: str) -> List[str]:
    s = norm_space(ans_raw)

    if qtype == "judge":
        if any(x in s for x in ["A", "正确", "对", "√"]):
            return ["A"]
        if any(x in s for x in ["B", "错误", "错", "×"]):
            return ["B"]
        return []

    # Normalize Chinese separators and keep only letters.
    s = s.replace("、", "").replace("，", "").replace(",", "").replace(" ", "")
    letters = [ch for ch in s if "A" <= ch <= "Z"]
    letters = sorted(set(letters))
    return letters


def parse_options_and_question(body_lines: List[str], qtype: str) -> Tuple[str, List[Dict[str, str]]]:
    if qtype == "judge":
        text = " ".join(body_lines)
        text = re.sub(r"A[\.．、]\s*正确\s*B[\.．、]\s*错误", "", text)
        text = re.sub(r"A\s*正确\s*B\s*错误", "", text)
        question = norm_space(text)
        opts = [
            {"key": "A", "text": "正确"},
            {"key": "B", "text": "错误"},
        ]
        return question, opts

    q_lines: List[str] = []
    options: List[Dict[str, str]] = []
    current_opt: Optional[Dict[str, str]] = None

    for line in body_lines:
        m = OPTION_START_RE.match(line)
        if m:
            if current_opt is not None:
                current_opt["text"] = norm_space(current_opt["text"])
                options.append(current_opt)
            current_opt = {"key": m.group(1), "text": m.group(2).strip()}
        else:
            if current_opt is not None:
                current_opt["text"] += " " + line
            else:
                q_lines.append(line)

    if current_opt is not None:
        current_opt["text"] = norm_space(current_opt["text"])
        options.append(current_opt)

    question = norm_space(" ".join(q_lines))
    return question, options


def keys_continuous(options: List[Dict[str, str]]) -> bool:
    keys = [o.get("key", "") for o in options]
    if not keys:
        return False
    expected = [chr(ord("A") + i) for i in range(len(keys))]
    return keys == expected


def validate_record(rec: Dict) -> Optional[str]:
    qtype = rec["type"]
    answer = rec["answer"]
    options = rec["options"]

    if not rec["question"]:
        return "question 为空"

    if not keys_continuous(options):
        return "options key 非连续大写"

    option_keys = {o["key"] for o in options}
    if not answer:
        return "answer 为空"
    if any(a not in option_keys for a in answer):
        return "answer 存在不在 options 中的 key"

    if qtype == "single" and len(answer) != 1:
        return "single 的 answer.length 必须为 1"
    if qtype == "multiple" and len(answer) < 2:
        return "multiple 的 answer.length 必须 >= 2"
    if qtype == "judge" and answer not in (["A"], ["B"]):
        return "judge 的 answer 只能是 ['A'] 或 ['B']"

    return None


def transform(raw: RawQuestion) -> Tuple[Optional[Dict], Optional[Dict]]:
    raw_text = ""
    if raw.lines:
        raw_text = f"{raw.no}.{raw.lines[0]}"
        if len(raw.lines) > 1:
            raw_text += "\n" + "\n".join(raw.lines[1:])
    raw_text = raw_text.strip()

    joined = "\n".join(raw.lines)
    ans_match = ANSWER_RE.search(joined)
    if not ans_match:
        err = {
            "no": raw.no,
            "type": raw.qtype,
            "sourcePage": raw.source_page,
            "reason": "未找到正确答案",
            "rawText": raw_text,
        }
        return None, err

    ans_raw = ans_match.group(1).strip()

    analysis = ""
    ana_match = ANALYSIS_RE.search(joined)
    if ana_match:
        analysis = norm_space(ana_match.group(1))

    # Remove answer/analysis lines before parsing question/options.
    body_lines: List[str] = []
    for line in raw.lines:
        if re.search(r"正确答案[：:]", line):
            continue
        if re.search(r"(?:解析|答案解析)[：:]", line):
            continue
        body_lines.append(line)

    question, options = parse_options_and_question(body_lines, raw.qtype)
    answer = parse_answer_tokens(ans_raw, raw.qtype)

    rec = {
        "id": "",
        "no": raw.no,
        "type": raw.qtype,
        "question": question,
        "options": options,
        "answer": answer,
        "analysis": analysis,
        "category": TYPE_META[raw.qtype]["category"],
        "chapter": "",
        "caseGroupId": "",
        "sourcePage": raw.source_page,
        "rawText": raw_text,
    }

    reason = validate_record(rec)
    if reason:
        err = {
            "no": raw.no,
            "type": raw.qtype,
            "sourcePage": raw.source_page,
            "reason": reason,
            "rawText": raw_text,
        }
        return None, err

    return rec, None


def main() -> None:
    cwd = Path('.')
    pdf_files = list(cwd.glob('*.pdf'))
    if not pdf_files:
        raise SystemExit('No PDF file found in current directory.')

    pdf_path = pdf_files[0]
    raw_questions = build_raw_questions(pdf_path)

    by_type: Dict[str, List[Dict]] = {"single": [], "multiple": [], "judge": [], "case": []}
    errors: List[Dict] = []

    for rq in raw_questions:
        rec, err = transform(rq)
        if err:
            errors.append(err)
            continue
        by_type[rec["type"]].append(rec)

    # assign IDs per type
    for qtype, items in by_type.items():
        for i, item in enumerate(items, start=1):
            item["id"] = f"{qtype}-{i:04d}"

    all_questions: List[Dict] = []
    for qtype in ["single", "multiple", "judge", "case"]:
        all_questions.extend(by_type[qtype])

    # Write files
    (cwd / 'single.json').write_text(json.dumps(by_type["single"], ensure_ascii=False, indent=2), encoding='utf-8')
    (cwd / 'multiple.json').write_text(json.dumps(by_type["multiple"], ensure_ascii=False, indent=2), encoding='utf-8')
    (cwd / 'judge.json').write_text(json.dumps(by_type["judge"], ensure_ascii=False, indent=2), encoding='utf-8')
    (cwd / 'case.json').write_text(json.dumps(by_type["case"], ensure_ascii=False, indent=2), encoding='utf-8')
    (cwd / 'question-bank.json').write_text(json.dumps(all_questions, ensure_ascii=False, indent=2), encoding='utf-8')
    (cwd / 'errors.json').write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding='utf-8')

    summary = {
        "pdf": str(pdf_path),
        "totalRawQuestions": len(raw_questions),
        "totalParsed": len(all_questions),
        "totalErrors": len(errors),
        "counts": {
            "single": len(by_type["single"]),
            "multiple": len(by_type["multiple"]),
            "judge": len(by_type["judge"]),
            "case": len(by_type["case"]),
        },
        "validationRules": {
            "single": "answer.length == 1",
            "multiple": "answer.length >= 2",
            "judge": "answer in [['A'], ['B']]",
            "optionsKeys": "连续且大写",
            "answerInOptions": True,
            "questionNotEmpty": True,
        },
    }
    (cwd / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
