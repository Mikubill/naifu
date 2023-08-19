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

def clean_limited_shuffle_dropout_processor(prompt: str) -> (str, bool):
    """
    cleaning:
        - remove tags that are subsets of other tags
        - remove noisy symbols and urls from human inputs

    limited shuffling:
        - shuffle tags in the prompt, excluding the starting sentence
        - shuffle begins at the first <SHUFFLE_START_OFFSET>th tag
        - shuffles with a limited shuffle distance of <MAX_SHUFFLE_OFFSET>

    random dropout:
        - dropout images with synthetic tags <SYNTHETIC_MARKER> with a probability of <SYNTHETIC_DROPOUT_RATE>

    :param prompt: raw prompt
    :return: (prompt:str, skip_image:bool)
    """
    SEPARATOR = ","  # separator between tags
    IS_SENTENCE_THRES = 4  # minimum space-separated parts of the starting prompt to be seen as sentence
    SHUFFLE_START_OFFSET = 2  # start shuffling at first-n <separator> separated part
    MAX_SHUFFLE_OFFSET = 4  # maximum distance of shuffle from the original position
    SYNTHETIC_MARKER = "<|generated|>"  # marker for synthetic images
    SYNTHETIC_DROPOUT_RATE = 0.6  # probability of dropping synthetic images

    import re

    def _find_subset_tags(tags_list: list) -> set:
        subset_tags = set()
        for i, tag1 in enumerate(tags_list):
            for j, tag2 in enumerate(tags_list):
                if i != j and tag1.strip() in tag2:
                    subset_tags.add(tag1)
                    break
        return subset_tags

    def _limited_shuffle(tags: list[str], max_offset: int) -> list[str]:
        shuffled_tags = tags.copy()
        n = len(tags)
        swapped_indices = set()  # swapped indices

        for i in range(n):
            if i in swapped_indices:  # swapped already
                continue

            # set the left and right bound of the offset
            left_bound = max(i - max_offset, 0)
            right_bound = min(i + max_offset, n - 1)

            # Select a random index within the offset range
            swap_index = random.randint(left_bound, right_bound)

            # Not swapping with itself and not swapping with an already swapped index
            if swap_index != i and swap_index not in swapped_indices:
                # Randomly swaps the tags & mark the indexes as swapped
                shuffled_tags[i], shuffled_tags[swap_index] = shuffled_tags[swap_index], shuffled_tags[i]
                swapped_indices.add(i)
                swapped_indices.add(swap_index)

        return shuffled_tags

    if not prompt:
        return prompt, False

    # Marker exists and is not a empty string
    if SYNTHETIC_MARKER and SYNTHETIC_MARKER in prompt:
        if random.random() < SYNTHETIC_DROPOUT_RATE:
            return prompt, True

    fixed_prompt = prompt.replace(SYNTHETIC_MARKER, "").strip()

    # Remove url
    fixed_prompt = re.sub(r"<http\S+>", "", fixed_prompt).strip()

    # Remove common webui tags
    STRIP_KEYS = ("BREAK", "&", "$", "#", "**", "((", "))", "\\", "\"")
    for key in STRIP_KEYS:
        fixed_prompt = fixed_prompt.replace(key, "")

    # Fix common punctuations
    REPLACEMENT_MAP = [("\n", ", "), (" ,", ","), (" .", "."), (" :", ":"), (" ;", ";"), ("  ", " "), ("、", ","),
                       ("|", ","), ]

    for _from, _to in iter(REPLACEMENT_MAP):
        fixed_prompt = fixed_prompt.replace(_from, _to)

    # Remove double spaces
    fixed_prompt = re.sub(r"\s+", " ", fixed_prompt).strip()

    # Splitting prompt by separator
    parts = fixed_prompt.split(SEPARATOR)
    parts = [part.strip() for part in parts]

    # Finding the index of the first non-sentence
    sentence_index = next((i for i, part in enumerate(parts) if len(part.split()) <= IS_SENTENCE_THRES), len(parts))

    # Keep the first <sentence_index> parts as it is
    preserved, to_shuffle = parts[:sentence_index], parts[sentence_index:]

    subset_tags = _find_subset_tags(to_shuffle)
    tags_to_keep = [tag for tag in to_shuffle if tag not in subset_tags]

    # Shuffling the remaining tags
    kept_tags = tags_to_keep[:SHUFFLE_START_OFFSET]
    shuffled_tags = _limited_shuffle(tags_to_keep[SHUFFLE_START_OFFSET:], max_offset=MAX_SHUFFLE_OFFSET)
    out = ", ".join(preserved + kept_tags + shuffled_tags)

    return out, False