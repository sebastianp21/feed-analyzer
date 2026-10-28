"""Text cleanup shared by training (model/prepare_data.py) and the server.

Our datasets mark usernames and links in different ways:
  TweetTopic: {{USERNAME}}, {{URL}}, {@WA Police Force@}
  TweetEval:  @user, URL
  Webis / X:  real @handles and https://... links
If we left these as-is, the model could learn "contains {{URL}} -> news"
instead of learning the wording. So every source, and every live post,
goes through normalize_text() and ends up in one format: "@user" and "http".
"""

import html
import re

TWEETTOPIC_ENTITY = re.compile(r"\{@[^{}]*@\}")  # {@Some Name@}
URL = re.compile(r"(https?://|www\.)\S+")
MENTION = re.compile(r"@\w+")
WHITESPACE = re.compile(r"\s+")
LEADING_MENTIONS = re.compile(r"^(@user\s*)+")


def normalize_text(text):
    text = html.unescape(text)  # "&amp;" -> "&" (common in tweet dumps)
    text = text.replace("{{USERNAME}}", "@user").replace("{{URL}}", "http")
    text = TWEETTOPIC_ENTITY.sub("@user", text)
    text = URL.sub("http", text)
    text = MENTION.sub("@user", text)
    text = re.sub(r"\bURL\b", "http", text)  # TweetEval's link placeholder
    text = WHITESPACE.sub(" ", text).strip()
    # Replies start with "@user @user ..." in the datasets, but X doesn't show those
    # mentions in the post text. Remove them so the model can't learn "reply = outrage".
    return LEADING_MENTIONS.sub("", text)
