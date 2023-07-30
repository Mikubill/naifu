import random
import json

# return format: (prompt, skip_image)

def shuffle_prompts(prompt, separator=",", start="Tags:", offset=3):
    try:
        index = prompt.index(start)
    except ValueError:
        start = ""
        index = 0
        
    if len(prompt) == 0:
        return prompt, False
        
    subprompt = prompt[index + len(start):] 
    tags = subprompt.split(separator)
    preserved, tags = tags[:offset], tags[offset:]
    
    random.shuffle(tags) 
    out = prompt[:index] + start + separator.join(preserved + tags)
    return out, False
    
def nai_tag_processor(tags, min_tags=24, max_tags=72, type_dropout=0.75, keep_important=1.00, keep_jpeg_artifacts=True, sort_tags=False) -> str:
    
    if isinstance(tags, str):
        tags = tags.replace(",", " ").split(" ")
        tags = [tag.strip() for tag in tags if tag != ""]
    final_tags = {}

    tag_dict = {tag: True for tag in tags}
    pure_tag_dict = {tag.split(":", 1)[-1]: tag for tag in tags}
    for bad_tag in ["absurdres", "highres", "translation_request", "translated", "commentary", "commentary_request", "commentary_typo", "character_request", "bad_id", "bad_link", "bad_pixiv_id", "bad_twitter_id", "bad_tumblr_id", "bad_deviantart_id", "bad_nicoseiga_id", "md5_mismatch", "cosplay_request", "artist_request", "wide_image", "author_request", "artist_name"]:
        if bad_tag in pure_tag_dict:
            del tag_dict[pure_tag_dict[bad_tag]]

    if "rating:questionable" in tag_dict or "rating:explicit" in tag_dict or "nsfw" in tag_dict:
        final_tags["nsfw"] = True

    base_chosen = []
    skip_image = False
    # counts = [0]
    
    for tag in tag_dict.keys():
        # For danbooru tags.
        parts = tag.split(":", 1)
        if parts[0] in ["artist", "copyright", "character"] and random.random() < keep_important:
            base_chosen.append(tag)
        if len(parts[-1]) > 1 and parts[-1][0] in ["1", "2", "3", "4", "5", "6"] and parts[-1][1:] in ["boy", "boys", "girl", "girls"]:
            base_chosen.append(tag)
        if parts[-1] in ["6+girls", "6+boys", "bad_anatomy", "bad_hands"]:
            base_chosen.append(tag)

    tag_count = min(random.randint(min_tags, max_tags), len(tag_dict.keys()))
    base_chosen_set = set(base_chosen)
    chosen_tags = base_chosen + [tag for tag in random.sample(list(tag_dict.keys()), tag_count) if tag not in base_chosen_set]
    if sort_tags:
        chosen_tags = sorted(chosen_tags)

    for tag in chosen_tags:
        tag = tag.replace(",", "").replace("_", " ")
        if random.random() < type_dropout:
            if tag.startswith("artist:"):
                tag = tag[7:]
            elif tag.startswith("copyright:"):
                tag = tag[10:]
            elif tag.startswith("character:"):
                tag = tag[10:]
            elif tag.startswith("general:"):
                tag = tag[8:]
        if tag.startswith("meta:"):
            tag = tag[5:]
        final_tags[tag] = True

    return "Tags: " + ", ".join(list(final_tags.keys())), False

def tags_only(input:str):
    inp = json.loads(input)
    tags = inp["tags"]
    characters = inp["characters"]
    ofa = inp["ofa"]
    gpt = inp["gpt"]
    return ','.join(tags).replace('_', ' '), False