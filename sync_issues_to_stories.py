import os
import requests

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")  # Make sure this is set
REPO = "katrin-ibrahim/Collaborative-Writing-with-LLM-based-Agents"
STORIES_FILE = "stories.md"

HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
}


def fetch_issues():
    url = f"https://api.github.com/repos/{REPO}/issues?state=open&per_page=100"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    return response.json()


def format_issue(issue):
    labels = [label["name"] for label in issue.get("labels", [])]
    estimate = next((l.split(":")[1] for l in labels if l.startswith("estimate:")), "0")
    type_ = next(
        (l for l in labels if l in {"bug", "feature", "technical-task", "test"}),
        "technical-task",
    )
    priority = next(
        (l for l in labels if l in {"low", "medium", "high", "critical"}), "medium"
    )

    return f"""\
### {issue['title']}

- ID: {issue['title'].split(']')[0].strip('[')}
- Type: {type_}
- Priority: {priority}
- Estimate: {estimate}
- Description: {issue['body'] or ''}
"""


def append_all_issues():
    print("📥 Fetching all open issues from GitHub...")
    issues = fetch_issues()

    with open(STORIES_FILE, "a") as f:
        for issue in issues:
            f.write("\n" + format_issue(issue))
            print(f"✓ Added: {issue['title']}")


if __name__ == "__main__":
    append_all_issues()
