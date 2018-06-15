import os
import sys
import time
import copy
import collections

import numpy as np

start_time = time.time()

# detailed: hierarchical distance between classes
detailed = len(sys.argv) > 2

## supported datasets 
# ImageNet = ILSVRC2012 + ImageNet2011: labels in the order of wnid, known first
# AWA, CUB: labels in the order of class name, known first
supported_datasets = ['ImageNet', 'AWA', 'CUB']
if len(sys.argv) > 1:
    assert sys.argv[1] in supported_datasets, 'supported datasets: {s}'.format(s=str(supported_datasets))
    dataset = sys.argv[1]
else:
    dataset = 'ImageNet'
print('{time:8.3f} s; dataset: {dataset}'.format(time=time.time()-start_time, dataset=dataset))

taxonomy_path = 'taxonomy/{dataset}'.format(dataset=dataset)
if not os.path.isdir(taxonomy_path):
    os.makedirs(taxonomy_path)
    raise FileNotFoundError('put txt files in {s}'.format(s=taxonomy_path))

# ancestors to be skipped
if dataset == 'AWA':
    ancestors_skip = ['n01861778', 'n01317541'] # mammal, domestic animal 
elif dataset == 'CUB':
    ancestors_skip = ['n01471682'] # vertebrate
else:
    ancestors_skip = []

def merge_wnids(wnid, ch_=None):
    if wnid in wnid_roots:
        if ch_ is None:
            assert len(wnid_children[wnid]) == 1, \
                   '{time:8.3f} s; specify new root in the second arg'.format(time=time.time()-start_time)
            ch_ = list(wnid_children[wnid])[0]
        print('{time:8.3f} s; root is changed from {wnid_roots} to {wnid_root_new}' \
              .format(time=time.time()-start_time, wnid_roots=wnid_roots, wnid_root_new=ch_))
        wnid_roots.remove(wnid)
        wnid_roots.append(ch_)
        for ch in (wnid_children[wnid] - {ch_}):
            merge_wnids(ch)
    wnid_group = wnid_groups.pop(wnid)
    chs = wnid_children.pop(wnid)
    pas = wnid_parents.pop(wnid)
    for pa in pas:
        wnid_groups[pa].update(wnid_group)
        wnid_children[pa].remove(wnid)
        wnid_children[pa].update(chs)
    for ch in chs:
        wnid_groups[ch].update(wnid_group)
        wnid_parents[ch].remove(wnid)
        wnid_parents[ch].update(pas)

# wnid parent to children and child to parents
wnid_is_a_path = 'taxonomy/wordnet.is_a.txt'
wnid_is_a_list = open(wnid_is_a_path, 'r').read().strip().replace(' ', '\n').splitlines()
assert len(wnid_is_a_list) % 2 == 0, 'a parent wnid has no child?'
wnid_is_a_pairs = [wnid_is_a_list[0::2], wnid_is_a_list[1::2]]
wnid_parent_to_children = {wnid: set() for wnid in set(wnid_is_a_pairs[0])}
wnid_child_to_parents   = {wnid: set() for wnid in set(wnid_is_a_pairs[1])}
for i in range(len(wnid_is_a_list) // 2):
    pa, ch = wnid_is_a_pairs[0][i], wnid_is_a_pairs[1][i]
    wnid_parent_to_children[pa].add(ch)
    wnid_child_to_parents[ch].add(pa)
print('{time:8.3f} s; wnid_parent_to_children and wnid_child_to_parents'.format(time=time.time()-start_time))

if dataset == 'ImageNet':
    # leaf wnids in hierarchy
    wnids_leaf_path = '{taxonomy_path}/classes_known.txt'.format(taxonomy_path=taxonomy_path)
    wnids_leaf = open(wnids_leaf_path, 'r').read().strip().replace('\t', '\n').splitlines()[::2]
    
    # novel wnids
    min_num_images = 50
    wnids_novel_path = '{taxonomy_path}/classes_novel.txt'.format(taxonomy_path=taxonomy_path)
    wnids_novel_list = open(wnids_novel_path, 'r').read().strip().replace('\t', '\n').splitlines()
    wnids_novel, num_images_novel = wnids_novel_list[0::2], [int(n) for n in wnids_novel_list[1::2]]
    print('{time:8.3f} s; {num_wnids} novel class candidates' \
          .format(time=time.time()-start_time, num_wnids=len(wnids_novel)))
    wnids_novel = [wnid for k, wnid in enumerate(wnids_novel) if num_images_novel[k] >= min_num_images]
    print('{time:8.3f} s; {num_wnids} novel class candidates with >= {min_num_images} images' \
          .format(time=time.time()-start_time, num_wnids=len(wnids_novel), min_num_images=min_num_images))

elif dataset == 'AWA' or dataset == 'CUB':
    # leaf wnids in hierarchy
    words_leaf_path = '{taxonomy_path}/trainvalclasses.txt'.format(taxonomy_path=taxonomy_path)
    words_leaf = open(words_leaf_path, 'r').read().strip().splitlines()
    
    words_novel_path = '{taxonomy_path}/testclasses.txt'.format(taxonomy_path=taxonomy_path)
    words_novel = open(words_novel_path, 'r').read().strip().splitlines()
    
    # update wnid parent to children and child to parents
    # parents are in the order of the class id, not the line number
    words_all_path = '{taxonomy_path}/allclasses.txt'.format(taxonomy_path=taxonomy_path)
    words_all = open(words_all_path, 'r').read().strip().splitlines()
    wnids_all = ['n90000{id:03d}'.format(id=k+1) for k in range(len(words_all))]
    word_to_wnid_leaf = dict(zip(words_all, wnids_all))
    wnids_leaf = [word_to_wnid_leaf[word] for word in words_leaf]
    wnids_novel = [word_to_wnid_leaf[word] for word in words_novel]
    if dataset == 'AWA':
        wnid_parents_path = '{taxonomy_path}/awa_classes_offset_rev1.txt'.format(taxonomy_path=taxonomy_path)
        wnid_parents_leaf = dict(zip(wnids_all, open(wnid_parents_path, 'r').read().strip().splitlines()))
    else:
        synset_parents_path = '{taxonomy_path}/cub_classes_wordnet_rev1.txt'.format(taxonomy_path=taxonomy_path)
        synset_parents = open(synset_parents_path, 'r').read().strip().splitlines()
        from nltk.corpus import wordnet as wn
        wnid_parents_leaf = {word_to_wnid_leaf[word]:
                             'n{offset:08d}'.format(offset=wn.synset(synset_parents[k]).offset())
                             for k, word in enumerate(sorted(words_all))}
    for ch in wnid_parents_leaf:
        pa = wnid_parents_leaf[ch]
        assert wnid_child_to_parents.get(pa) is not None, \
            print('{ch}: no parent {pa} in the initial is-a relationship'.format(ch=ch, pa=pa))
        if wnid_parent_to_children.get(pa) is None:
            wnid_parent_to_children[pa] = {ch}
        else:
            wnid_parent_to_children[pa].add(ch)
        if wnid_child_to_parents.get(ch) is None:
            wnid_child_to_parents[ch] = {pa}
        else:
            wnid_child_to_parents[ch].add(pa)
    print('{time:8.3f} s; wnid_parent_to_children and wnid_child_to_parents update' \
          .format(time=time.time()-start_time))
else:
    raise NotImplementedError('unsupported dataset: {dataset}'.format(dataset=dataset))

# sanity check: words are related to their parents
if dataset == 'AWA' or dataset == 'CUB':
    # wnid to word
    wnid_to_word_path = 'taxonomy/words.txt'
    wnid_to_word_list = open(wnid_to_word_path, 'r').read().strip().replace('\t', '\n').splitlines()
    assert len(wnid_to_word_list) % 2 == 0, 'a wnid has no word?'
    wnid_to_word = dict(zip(wnid_to_word_list[0::2], wnid_to_word_list[1::2]))
    print('wnid_to_word; {time:8.3f} s'.format(time=time.time()-start_time))
    # update wnid to word
    for k, word in enumerate(words_leaf):
        wnid_to_word[wnids_leaf[k]] = word
    for k, word in enumerate(words_novel):
        wnid_to_word[wnids_novel[k]] = word
    for wnid in sorted(wnids_leaf + wnids_novel):
        for pa in wnid_child_to_parents[wnid]:
            print('{word} - {word_pa}'.format(word=wnid_to_word[wnid], word_pa=wnid_to_word[pa]))

# wnids in the raw taxonomy
wnids_raw = set(wnids_leaf)
num_leaves = len(wnids_raw)
wnids_visited = set()
wnid_roots = set()
while wnids_raw != wnids_visited:
    for wnid in (wnids_raw - wnids_visited):
        if wnid_child_to_parents.get(wnid) is None:
            wnid_roots.add(wnid)
        else:
            pas = wnid_child_to_parents[wnid]
            pas.difference_update(ancestors_skip)
            if pas:
                wnids_raw.update(pas)
            else:
                wnid_roots.add(wnid)
        wnids_visited.add(wnid)
    print('{time:8.3f} s; {num_wnids_visited:4d}/{num_wnids:4d} build initial wnid list' \
          .format(time=time.time()-start_time, num_wnids_visited=len(wnids_visited), num_wnids=len(wnids_raw)))
wnid_roots = sorted(wnid_roots)
print('{time:8.3f} s; roots: {roots}'.format(time=time.time()-start_time, roots=str(wnid_roots)))

# add a global root if there are multiple roots
if len(wnid_roots) > 1:
    wnid_root = '_root_'
    wnids_raw.append(wnid_root)
    for wnid in wnid_roots:
        wnid_parent_to_children[wnid_root].add(wnid)
        wnid_child_to_parents[wnid].add(wnid_root)
    wnid_roots = [wnid_root]
elif len(wnid_roots) == 0:
    raise AssertionError('no root')

# sort raw wnids: [leaves, supers in ascend order]
wnids_super = sorted(wnids_raw.difference(wnids_leaf))
wnids_raw = copy.deepcopy(wnids_leaf)
wnids_raw.extend(wnids_super)

# wnid_parents
wnid_parents = {wnid: [] for wnid in wnids_raw}
for wnid in wnids_raw:
    if wnid_child_to_parents.get(wnid) is not None:
        wnid_parents[wnid] = wnid_child_to_parents[wnid].intersection(wnids_raw)
print('{time:8.3f} s; wnid_parents'.format(time=time.time()-start_time))

# wnid_children
wnid_children = {wnid: [] for wnid in wnids_raw}
for wnid in wnids_raw:
    if wnid_parent_to_children.get(wnid) is not None:
        wnid_children[wnid] = wnid_parent_to_children[wnid].intersection(wnids_raw)
print('{time:8.3f} s; wnid_children'.format(time=time.time()-start_time))

# wnid_groups
wnid_groups = collections.OrderedDict((wnid, {wnid}) for wnid in wnids_raw)

# remove classes with only one child; order matters
wnid_queue = copy.deepcopy(wnids_leaf)
while wnid_queue:
    wnid, wnid_queue = wnid_queue[0], wnid_queue[1:]
    if wnid_groups.get(wnid) is None:
        continue
    for pa in sorted(wnid_parents[wnid]):
        if pa not in wnid_queue:
            wnid_queue.append(pa)
    if len(wnid_children[wnid]) == 1:
        merge_wnids(wnid)
print('{time:8.3f} s; removed classes with only one child; {num_wnids} remaining' \
      .format(time=time.time()-start_time, num_wnids=len(wnid_groups)))

# wnid descendant leaves
wnid_de_leaves = {wnid: {wnid} for wnid in wnids_leaf}
wnid_de_leaves.update({wnid: set() for wnid in set(wnid_groups).difference(wnids_leaf)})
wnid_queue = set(wnids_leaf)
while wnid_queue:
    wnid = wnid_queue.pop()
    for pa in wnid_parents[wnid]:
        wnid_de_leaves[pa].add(wnid)
        wnid_de_leaves[pa].update(wnid_de_leaves[wnid])
        wnid_de_leaves[pa].intersection_update(wnids_leaf)
        wnid_queue.add(pa)
print('{time:8.3f} s; wnid_de_leaves'.format(time=time.time()-start_time))

# remove classes sharing the same leaves with its child; order matters
wnid_queue = copy.deepcopy(wnids_leaf)
while wnid_queue:
    wnid, wnid_queue = wnid_queue[0], wnid_queue[1:]
    if wnid_groups.get(wnid) is None:
        continue
    for pa in sorted(wnid_parents[wnid]):
        if pa not in wnid_queue:
            wnid_queue.append(pa)
    for ch_ in sorted(wnid_children[wnid]):
        if wnid_groups.get(ch_) is None:
            continue
        if wnid_de_leaves[wnid] == wnid_de_leaves[ch_]:
            merge_wnids(wnid, ch_)
            print('{time:8.3f} s; {wnid} and {ch} have the same leaves' \
                  .format(time=time.time()-start_time, wnid=wnid, ch=ch_))
            break
print('{time:8.3f} s; removed classes sharing the same leaves with its child; {num_wnids} remaining' \
      .format(time=time.time()-start_time, num_wnids=len(wnid_groups)))

print('{time:8.3f} s; a group has  max {max_wnids:4d} wnids' \
      .format(time=time.time()-start_time, max_wnids=max([len(g) for g in wnid_groups.values()])))
wnid_counter = collections.Counter([wnid for wnid_group in wnid_groups.values() for wnid in wnid_group])
print('{time:8.3f} s; a wnid is in max {max_wnids:4d} groups' \
      .format(time=time.time()-start_time, max_wnids=max(wnid_counter.values())))

assert len(wnid_roots) == 1, 'multiple roots'
wnid_root = wnid_roots[0]

# essential wnids
wnids = copy.deepcopy(wnids_leaf) # sorted(wnids_leaf)
wnids.extend(sorted(set(wnid_groups).difference(wnids_leaf)))
assert wnids == list(wnid_groups), 'wnid groups inconsistent'
assert set(wnids) == set(wnid_parents), 'wnid_parents.keys() has removed wnids?'
assert set(wnids) == set(wnid_children), 'wnid_children.keys() has removed wnids?'
wnids = np.array(wnids) # for np.nonzero

# sort elements in wnid groups, parents, children
wnid_to_index = {wnid: k for k, wnid in enumerate(wnids)}
wnid_groups = {wnid: sorted(wnid_groups[wnid]) for wnid in wnids}
wnid_parents = {wnid: sorted(wnid_parents[wnid], key=lambda pa: wnid_to_index[pa]) for wnid in wnids}
wnid_children = {wnid: sorted(wnid_children[wnid], key=lambda ch: wnid_to_index[ch]) for wnid in wnids}
wnid_to_group = {wnid: [] for wnid in wnids_raw}
for wnid_front in wnids:
    for wnid_back in wnid_groups[wnid_front]:
        wnid_to_group[wnid_back].append(wnid_front)

# check whether taxonomy or DAG
taxonomy_type = 'DAG' if max([len(pa) for pa in wnid_parents.values()]) > 1 else 'tree'
print('\n{time:8.3f} s; this taxonomy is {type}\n'.format(time=time.time()-start_time, type=taxonomy_type))

# wnid_ancestors
wnid_ancestors = {wnid: {wnid: 0} for wnid in wnids}
wnid_queue = {wnid_root}
wnid_depths = {wnid_root: 0} # deepest depth; dist_mat[num_leaves] for shallowest depth
while wnid_queue:
    wnid = wnid_queue.pop()
    for ch in wnid_children[wnid]:
        if wnid_depths.get(ch) is None or wnid_depths[ch] < wnid_depths[wnid] + 1:
            wnid_depths[ch] = wnid_depths[wnid] + 1
        for an in wnid_ancestors[wnid]:
            old_dist = wnid_ancestors[ch].get(an)
            dist = wnid_ancestors[wnid][an]
            if old_dist is None or old_dist > dist+1:
                wnid_ancestors[ch][an] = dist+1
        old_dist = wnid_ancestors[ch].get(wnid)
        if old_dist is None or old_dist > 1:
            wnid_ancestors[ch][wnid] = 1
        wnid_queue.add(ch)
print('{time:8.3f} s; wnid_ancestors'.format(time=time.time()-start_time))

# wnid_descendants
wnid_descendants = {wnid: {wnid: 0} for wnid in wnids}
wnid_queue = set(wnids_leaf)
wnid_heights = {wnid: 0 for wnid in wnids}
while wnid_queue:
    wnid = wnid_queue.pop()
    for pa in wnid_parents[wnid]:
        if wnid_heights.get(pa) is None or wnid_heights[pa] < wnid_heights[wnid] + 1:
            wnid_heights[pa] = wnid_heights[wnid] + 1
        for de in wnid_descendants[wnid]:
            old_dist = wnid_descendants[pa].get(de)
            dist = wnid_descendants[wnid][de]
            if old_dist is None or old_dist > dist+1:
                wnid_descendants[pa][de] = dist+1
        old_dist = wnid_descendants[pa].get(wnid)
        if old_dist is None or old_dist > 1:
            wnid_descendants[pa][wnid] = 1
        wnid_queue.add(pa)
print('{time:8.3f} s; wnid_descendants'.format(time=time.time()-start_time))

# wnid_ancestors_hop
wnid_ancestors_hop = {wnid: dict() for wnid in wnids}
for wnid in wnid_ancestors:
    ancestors_to_hop = wnid_ancestors[wnid]
    for an in ancestors_to_hop:
        if wnid_ancestors_hop[wnid].get(ancestors_to_hop[an]) is None:
            wnid_ancestors_hop[wnid][ancestors_to_hop[an]] = [an]
        else:
            wnid_ancestors_hop[wnid][ancestors_to_hop[an]].append(an)
    for hop in wnid_ancestors_hop[wnid]:
        wnid_ancestors_hop[wnid][hop].sort()
print('{time:8.3f} s; wnid_ancestors_hop'.format(time=time.time()-start_time))

# wnid_descendants_hop
wnid_descendants_hop = {wnid: dict() for wnid in wnids}
for wnid in wnid_descendants:
    descendants_to_hop = wnid_descendants[wnid]
    for de in descendants_to_hop:
        if wnid_descendants_hop[wnid].get(descendants_to_hop[de]) is None:
            wnid_descendants_hop[wnid][descendants_to_hop[de]] = [de]
        else:
            wnid_descendants_hop[wnid][descendants_to_hop[de]].append(de)
    for hop in wnid_descendants_hop[wnid]:
        wnid_descendants_hop[wnid][hop].sort()
print('{time:8.3f} s; wnid_descendants_hop'.format(time=time.time()-start_time))

# is_parent_mat [pa[k], k] == [k, ch[k]]
is_parent_mat = np.zeros([len(wnids), len(wnids)], dtype=bool)
for k, wnid in enumerate(wnids):
    for pa in wnid_parents[wnid]:
        is_parent_mat[(pa == wnids).nonzero()[0], k] = True
print('{time:8.3f} s; is_parent_mat'.format(time=time.time()-start_time))

is_parent_mat_2 = np.zeros([len(wnids), len(wnids)], dtype=bool)
for k, wnid in enumerate(wnids):
    for ch in wnid_children[wnid]:
        is_parent_mat_2[k, (ch == wnids).nonzero()[0]] = True
print('{time:8.3f} s; is_parent_mat_2'.format(time=time.time()-start_time))

consistency = '' if (is_parent_mat == is_parent_mat_2).all() else '"not" '
print('{time:8.3f} s; parents and children are {c}consistent' \
      .format(time=time.time()-start_time, c=consistency))

# is_ancestor_mat [an[k], k] == [k, de[k]]
is_ancestor_mat = np.zeros([len(wnids), len(wnids)], dtype=bool)
for k, wnid in enumerate(wnids):
    for an in wnid_ancestors[wnid]:
        is_ancestor_mat[(an == wnids).nonzero()[0], k] = True
print('{time:8.3f} s; is_ancestor_mat'.format(time=time.time()-start_time))

is_ancestor_mat_2 = np.zeros([len(wnids), len(wnids)], dtype=bool)
for k, wnid in enumerate(wnids):
    for de in wnid_descendants[wnid]:
        is_ancestor_mat_2[k, (de == wnids).nonzero()[0]] = True
print('{time:8.3f} s; is_ancestor_mat_2'.format(time=time.time()-start_time))

consistency='' if (is_ancestor_mat == is_ancestor_mat_2).all() else '"not" '
print('{time:8.3f} s; ancestors and descendants are {c}consistent' \
      .format(time=time.time()-start_time, c=consistency))

if detailed:
    # dist_mat, dist_to_lca_mat
    MAX_DIST = 127
    dist_mat = MAX_DIST*np.ones([len(wnids), len(wnids)], dtype=np.int8)
    dist_to_lca_mat, dist_to_lca_mat_2 = copy.deepcopy(dist_mat), copy.deepcopy(dist_mat)
    for i, wnid_i in enumerate(wnids):
        for j, wnid_j in enumerate(wnids):
            dist = dist_to_lca_i = dist_to_lca_j = MAX_DIST
            for common_wnid in list(set(wnid_ancestors[wnid_i]).intersection(wnid_ancestors[wnid_j])):
                new_dist_to_lca_i = wnid_ancestors[wnid_i][common_wnid]
                new_dist_to_lca_j = wnid_ancestors[wnid_j][common_wnid]
                new_dist = new_dist_to_lca_i + new_dist_to_lca_j
                if dist > new_dist:
                    dist = new_dist
                if dist_to_lca_i > new_dist_to_lca_i:
                    dist_to_lca_i = new_dist_to_lca_i
                if dist_to_lca_j > new_dist_to_lca_j:
                    dist_to_lca_j = new_dist_to_lca_j
            dist_mat[i,j] = dist
            dist_to_lca_mat[i,j] = dist_to_lca_i
            dist_to_lca_mat_2[j,i] = dist_to_lca_j
    print('{time:8.3f} s; dist_mat and dist_to_lca_mat'.format(time=time.time()-start_time))
    consistency='' if (dist_mat == dist_mat.T).all() else '"not" '
    print('{time:8.3f} s; dist_mat is {c}consistent' \
          .format(time=time.time()-start_time, c=consistency))
    consistency='' if (dist_to_lca_mat == dist_to_lca_mat_2).all() else '"not" '
    print('{time:8.3f} s; dist_to_lca_mat is {c}consistent' \
          .format(time=time.time()-start_time, c=consistency))
    
    # num_ca_mat 
    num_ca_mat = np.zeros([len(wnids), len(wnids)], dtype=np.int8)
    for i, wnid_i in enumerate(wnids):
        for j, wnid_j in enumerate(wnids):
            num_ca_mat[i,j] = len(set(wnid_ancestors[wnid_i]).intersection(wnid_ancestors[wnid_j]))
    print('{time:8.3f} s; num_ca_mat'.format(time=time.time()-start_time))
    consistency='' if (num_ca_mat.T == num_ca_mat).all() else '"not" '
    print('{time:8.3f} s; num_ca_mat is {c}consistent' \
          .format(time=time.time()-start_time, c=consistency))
    
    # HP_mat, HF_mat
    HP_mat = np.zeros([len(wnids), len(wnids)])
    HR_mat, HF_mat = copy.deepcopy(HP_mat), copy.deepcopy(HP_mat)
    for i, wnid_i in enumerate(wnids): # pred
        for j, wnid_j in enumerate(wnids): # label
            num_common_ancestors = num_ca_mat[i,j] #- 1
            num_ancestors_i = len(wnid_ancestors[wnid_i]) #- 1
            num_ancestors_j = len(wnid_ancestors[wnid_j]) #- 1
            HP_mat[i,j] = num_common_ancestors / num_ancestors_i if num_ancestors_i > 0. else 0.
            HR_mat[i,j] = num_common_ancestors / num_ancestors_j if num_ancestors_j > 0. else 0.
            HF_mat[i,j] = 2.*num_common_ancestors / (num_ancestors_i+num_ancestors_j) if num_ancestors_i > 0. else 0.
    print('{time:8.3f} s; HP_mat and HF_mat'.format(time=time.time()-start_time))
    consistency='' if (HP_mat.T == HR_mat).all() else '"not" '
    print('{time:8.3f} s; HP_mat is {c}consistent' \
          .format(time=time.time()-start_time, c=consistency))
    consistency='' if (HF_mat == HF_mat.T).all() else '"not" '
    print('{time:8.3f} s; HF_mat is {c}consistent' \
          .format(time=time.time()-start_time, c=consistency))

# indexes
root = wnid_to_index[wnid_root]
parents = [[]]*len(wnids)
children = [[]]*len(wnids)
ancestors = [[]]*len(wnids)
descendants = [[]]*len(wnids)
ancestors_hop = [dict() for _ in range(len(wnids))]
descendants_hop = [dict() for _ in range(len(wnids))]
for k, wnid in enumerate(wnids):
    parents[k] = [wnid_to_index[pa] for pa in wnid_parents[wnid]]
    children[k] = [wnid_to_index[ch] for ch in wnid_children[wnid]]
    ancestors[k] = {wnid_to_index[an]: wnid_ancestors[wnid][an] for an in wnid_ancestors[wnid]}
    descendants[k] = {wnid_to_index[de]: wnid_descendants[wnid][de] for de in wnid_descendants[wnid]}
    for hop in wnid_ancestors_hop[wnid]:
        ancestors_hop[k][hop] = [wnid_to_index[an] for an in wnid_ancestors_hop[wnid][hop]]
    for hop in wnid_descendants_hop[wnid]:
        ancestors_hop[k][hop] = [wnid_to_index[de] for de in wnid_descendants_hop[wnid][hop]]
depths = [wnid_depths[wnid] for wnid in wnids]
heights = [wnid_heights[wnid] for wnid in wnids]
print('{time:8.3f} s; indexes'.format(time=time.time()-start_time))

# number of children and index slices for each super
num_children = [len(chs) for chs in children[num_leaves:]]
ch_slice = [0]
ch_slice.extend(np.cumsum(num_children).tolist())
print('{time:8.3f} s; num_children and ch_slice'.format(time=time.time()-start_time))


# novel to closest known ancestor
skip_leaf = True
skip_super = True
skip_under_leaf = True
skip_partially_under_leaf = skip_under_leaf and False
skip_counter = {'skip_leaf': 0, 'skip_super': 0, 'skip_under_leaf': 0, 'no_parents': 0}

wnid_ancestors_novel = {wnid: {wnid: 0} for wnid in wnids_novel}
wnid_novel_to_raw = {wnid: dict() for wnid in wnids_novel}
for wnid_novel in wnids_novel:
    if skip_leaf and wnid_novel in wnids_leaf:
        skip_counter['skip_leaf'] += 1
        continue
    elif skip_super and wnid_novel in wnids_raw:
        skip_counter['skip_super'] += 1
        continue
    wnid_queue = {wnid_novel}
    while wnid_queue:
        wnid = wnid_queue.pop()
        hop = wnid_ancestors_novel[wnid_novel][wnid]
        if wnid_child_to_parents.get(wnid) is None:
            skip_counter['no_parents'] += 1
            print('{time:8.3f} s; {wnid_novel}: {wnid} has no parents' \
                  .format(time=time.time()-start_time, wnid_novel=wnid_novel, wnid=wnid))
            continue
        for pa in wnid_child_to_parents[wnid]:
            # add to ancestor list
            if pa not in wnid_ancestors_novel[wnid_novel] or wnid_ancestors_novel[wnid_novel][pa] > hop+1:
                wnid_ancestors_novel[wnid_novel][pa] = hop+1
            # skip leaf
            if skip_under_leaf and pa in wnids_leaf:
                skip_counter['skip_under_leaf'] += 1
                continue
            # known: add to list
            elif pa in wnids_raw:
                if pa not in wnid_novel_to_raw[wnid_novel] or wnid_novel_to_raw[wnid_novel][pa] > hop+1:
                    wnid_novel_to_raw[wnid_novel][pa] = hop+1
            # novel: add to queue
            else:
                wnid_queue.add(pa)
print('{time:8.3f} s; {num_wnids:5d} find nearest super wnid' \
      .format(time=time.time()-start_time, num_wnids=len(wnids_novel)))
print('{time:8.3f} s; leaf: {skip_leaf:5d}, ' \
       .format(time=time.time()-start_time, skip_leaf=skip_counter['skip_leaf']) + \
       'super: {skip_super:5d}, under leaf: {skip_under_leaf:5d}, no parents: {no_parents:5d}' \
       .format(skip_super=skip_counter['skip_super'], skip_under_leaf=skip_counter['skip_under_leaf'],
               no_parents=skip_counter['no_parents']))
print('{time:8.3f} s; a wnid is in max {max_wnids:4d} known ancestors' \
      .format(time=time.time()-start_time, max_wnids=max([len(g) for g in wnid_novel_to_raw.values()])))

# novel to known
wnid_novel_to_wnid_known = collections.OrderedDict()
for wnid_novel in wnids_novel:
    if len(wnid_novel_to_raw[wnid_novel]) > 0:
        wnid_novel_to_wnid_known[wnid_novel] = dict()
        for wnid_back in wnid_novel_to_raw[wnid_novel]:
            for wnid_front in wnid_to_group[wnid_back]:
                if skip_partially_under_leaf and wnid_front in wnids_leaf:
                    continue
                if wnid_novel_to_wnid_known[wnid_novel].get(wnid_front) is None or \
                   wnid_novel_to_wnid_known[wnid_novel][wnid_front] > wnid_novel_to_raw[wnid_novel][wnid_back]:
                    wnid_novel_to_wnid_known[wnid_novel][wnid_front] = wnid_novel_to_raw[wnid_novel][wnid_back]
print('{time:8.3f} s; {num_wnids:5d} novel to known' \
      .format(time=time.time()-start_time, num_wnids=len(wnid_novel_to_wnid_known)))
print('{time:8.3f} s; a wnid is in max {max_wnids:4d} known groups' \
      .format(time=time.time()-start_time, max_wnids=max([len(g) for g in wnid_novel_to_wnid_known.values()])))

# filter unclassifiable novel
wnids_novel = list(wnid_novel_to_wnid_known)
wnid_ancestors_novel = {wnid: wnid_ancestors_novel[wnid] for wnid in wnids_novel}
wnid_novel_to_raw = {wnid: wnid_novel_to_raw[wnid] for wnid in wnids_novel}

# indexes
wnid_novel_to_known = {wnid: collections.OrderedDict() for wnid in wnid_novel_to_wnid_known}
# novel_to_known = [collections.OrderedDict() for _ in range(len(wnid_novel_to_wnid_known))]
for k, wnid in enumerate(wnid_novel_to_wnid_known):
    for an in sorted(wnid_novel_to_wnid_known[wnid], key=lambda an: (wnid_novel_to_wnid_known[wnid][an], an)):
        wnid_novel_to_known[wnid][wnid_to_index[an]] = wnid_novel_to_wnid_known[wnid][an]
        # novel_to_known[k][wnid_to_index[an]] = wnid_novel_to_wnid_known[wnid][an]
print('{time:8.3f} s; indexes novel'.format(time=time.time()-start_time))

# wnid to label
if dataset == 'AWA' or dataset == 'CUB':
    wnid_to_label = {'n90000{id:03d}'.format(id=k+1): k for k in range(num_leaves + len(wnids_novel))}
else:
    wnid_to_label = {wnids_leaf[k]: k for k in range(num_leaves)}
    wnid_to_label.update({wnids_novel[k]: k+num_leaves for k in range(len(wnids_novel))})

# original label to enumeration s.t. known first; mixed -> [known, novel]
# ImageNet does not require this; == identity mapping
# apply to AWA and CUB labels
label_enum_dict = {wnid_to_label[wnid]: wnid_to_index[wnid] for wnid in wnids_leaf}
label_enum_dict.update({wnid_to_label[wnid]: k+num_leaves for k, wnid in enumerate(wnids_novel)})
label_enum = [label_enum_dict[k] for k in range(len(label_enum_dict))]

# label enumeration to the closest classes
label_hnd = [collections.OrderedDict({k:0}) for k in range(num_leaves)]
label_hnd.extend([wnid_novel_to_known[wnid] for wnid in wnids_novel])

# original label to the closest class
label_zsl_dict = {wnid_to_label[wnid]: wnid_to_index[wnid] for wnid in wnids_leaf}
if skip_under_leaf and (not skip_partially_under_leaf):
    for wnid in wnids_novel:
        for l in wnid_novel_to_known[wnid]:
            if l >= num_leaves:
                label_zsl_dict[wnid_to_label[wnid]] = l
                break
    assert len(label_zsl_dict) == num_leaves + len(wnids_novel), 'some novel classes are under leaf only'
else:
    label_zsl_dict.update({wnid_to_label[wnid]: list(wnid_novel_to_known[wnid])[0] for wnid in wnids_novel})
label_zsl = [label_zsl_dict[k] for k in range(len(label_zsl_dict))]

print('{time:8.3f} s; label mapping'.format(time=time.time()-start_time))

save_me = {
    'wnids_leaf': wnids_leaf, 'wnids': wnids.tolist(), 'wnids_raw': wnids_raw,
    'wnid_groups': wnid_groups, 'wnid_to_group': wnid_to_group, 'wnid_to_index': wnid_to_index,
    'wnid_root': wnid_root, 'wnid_parents': wnid_parents, 'wnid_children': wnid_children,
    'wnid_ancestors': wnid_ancestors, 'wnid_descendants': wnid_descendants,
    'wnid_ancestors_hop': wnid_ancestors_hop, 'wnid_descendants_hop': wnid_descendants_hop,
    'wnid_depths': wnid_depths, 'wnid_heights': wnid_heights,
    'is_parent_mat': is_parent_mat, 'is_ancestor_mat': is_ancestor_mat,
    'root': root, 'parents': parents, 'children': children,
    'ancestors': ancestors, 'descendants': descendants,
    'ancestors_hop': ancestors_hop, 'descendants_hop': descendants_hop,
    'heights': heights, 'num_children': num_children, 'ch_slice': ch_slice,
    'wnids_novel': wnids_novel,
    'wnid_ancestors_novel': wnid_ancestors_novel, 'wnid_novel_to_raw': wnid_novel_to_raw,
    'wnid_novel_to_wnid_known': wnid_novel_to_wnid_known,
    'wnid_novel_to_known': wnid_novel_to_known,
    'label_enum': label_enum, 'label_hnd': label_hnd, 'label_zsl': label_zsl,
          }
if detailed:
    save_me.update({
        'dist_mat': dist_mat, 'dist_to_lca_mat': dist_to_lca_mat, 'num_ca_mat': num_ca_mat,
        'HP_mat': HP_mat, 'HF_mat': HF_mat
                   })

np.save(taxonomy_path + '/taxonomy.npy', save_me)

