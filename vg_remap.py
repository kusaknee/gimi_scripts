"""Modified vg_remap from https://github.com/SilentNightSound/GI-Model-Importer/blob/main/Tools/blender_vg_remap.txt.

This uses center of mass to find nearest groups, using the hypothesis that weights are probably centered around a bone in the middle.

# Instructions:

## Use case (basic): Map _SOURCE's vertex groups onto _DESTINATION

For example, if you want Nahida to use Klee's model, Nahida will be _SOURCE and Klee will be _DESTINATION.

1. Set _SOURCE to be the name of where canonical vertex group information should be read.

2. Set _DESTINATION to be the name of where

3. Run the script. It is advisable to make a copy of the mesh before running the script.

## Use case (more complex): Map _SOURCE's vertex groups onto _DESTINATION but with a transformation

For example, if you want to add a tail to Nahida, and want to copy the vertex groups from a part of her dress while keeping the tail
in her rear, you can make a copy of the tail and resize it to match that part of the dress.

The script can then do the comparison using that resized mesh, but rewrite the vertex groups in the original unscaled tail.

In this case, set the tail to be _DESTINATION, set the dress to be _SOURCE, and set the resized tail to be _REF_DEST.

Note that _REF_DEST must have the same vertex group names as _DESTINATION. Behavior for when _REF_DEST is not a copy of _DESTINATION is undefined.
"""
import bpy
import mathutils
from collections import defaultdict

# Where to read the vertex group data from.
_SOURCE = 'SOURCE'

# Where to write the vertex group data to.
_DESTINATION = 'DEST'

# If set, use this mesh as the reference for vertex group mapping. This is useful if you want to map creatively.
# Use this mesh as the destination for calculation purposes, but write to the destination for the final calculation results.
# For example, if you want to map a new tail 'TAIL' to head 'SOURCE_HEAD', you can duplicate the tail to get 'TAIL.001'.
# Then, you can resize the tail to match the hair parts you want to match in 'SOURCE_HEAD'.
_REF_DEST = ''

# Whether to take into account vertex weights when performing calculations. Use true if vertex groups aren't very symmetric.
_WEIGHTED = True

# If true, then automatically merge the vertex groups. Otherwise, multiple candidates for the same group will be labeled N.001, N.002, etc.
# Use true if mapping for final usage, use false if mapping to see how the results will look.
_MERGE = True

def merge_vgs(obj):
    """Merges similarly named vg groups together.

    A *much* quicker version of https://github.com/SilentNightSound/GI-Model-Importer/blob/main/Tools/blender_merge_vg.txt.

    That one is O(N^2M) (n=groups, m=vertices). This is O(NM).

    Args:
        obj: (bpy_types.Object) Object to parse through vertex groups and merge.
    """
    vg_index_to_name = {}
    vg_name_to_index = {}
    for vg in obj.vertex_groups:
        vg_index_to_name[vg.index] = vg.name
        vg_name_to_index[vg.name] = vg.index

    vg_to_vertex = defaultdict(lambda: defaultdict(float))
    for v in obj.data.vertices:
        for vg in v.groups:
            vg_name = vg_index_to_name.get(vg.group, '')
            # Reduce memory footprint by only storing vgs we are likely to pop.
            if '.' not in vg_name:
                continue

            main_name = vg_name.split('.')[0]

            vg_to_vertex[main_name][v.index] += vg.weight

    for vg, vidx_to_weight in vg_to_vertex.items():
        if vg not in vg_name_to_index:
            continue

        vg_ref = obj.vertex_groups[vg_name_to_index[vg]]
        for vidx, weight in vidx_to_weight.items():
            vg_ref.add([vidx], weight, 'ADD')

    for vg in obj.vertex_groups:
        if '.' in vg.name:
            obj.vertex_groups.remove(vg)


def calc_center_of_mass_per_vg(obj, weighted=False):
    """Calculates the center of mass per vertex group in the object.

    Args:
        obj: (bpy_types.Object) Object to read vertex groups from.
        weighted: (bool) Whether or not to take into account vertex weights when doing calculations.

    Returns:
        dict[str, (float, float, float)] - A map of vertex group name to the center of mass as an (x, y, z) coordinate.
    """
    index_to_name = {}
    # Vertices contain information only on vertex group index, not names.
    for vg in obj.vertex_groups:
        index_to_name[vg.index] = vg.name

    vertices_per_group = {}
    for v in obj.data.vertices:
        for vg in v.groups:
            if vg.weight == 0:
                continue

            if vg.group not in index_to_name:
                # Well, something went wrong.
                raise ValueError(f'Vertex has vertex group with index {vg.group} but this is not known.')

            vg_name = index_to_name[vg.group]

            if vg_name not in vertices_per_group:
                vertices_per_group[vg_name] = [[], []]

            weight = vg.weight if weighted else 1.0

            vv = obj.matrix_world @ v.co

            vertices_per_group[vg_name][0].append(vv)
            vertices_per_group[vg_name][1].append(weight)

    cm_per_group = {}

    for vg_name, (vertices, weights) in vertices_per_group.items():
        summed_weights = sum(weights)

        if summed_weights == 0:
            continue

        vw = zip(vertices, weights)

        x_center = 0.0
        y_center = 0.0
        z_center = 0.0

        for p, w in zip(vertices, weights):
            x_center += p.x * w
            y_center += p.y * w
            z_center += p.z * w

        x_center /= summed_weights
        y_center /= summed_weights
        z_center /= summed_weights

        cm_per_group[vg_name] = (x_center, y_center, z_center)

    return cm_per_group


def init_kdtree_vertex_to_cm(vg_to_vertex):
    """Converts a dictionary of vertex group center of masses to a KD tree.

    Args:
        vg_to_vertex: (dict[str, (float, float, float)]) Map of vertex groups to their center of masses.

    Returns:
        (mathutils.kdtree.KDTree, dict[int, str]) The first return is a KDTree. The second is a map of the
            kdtree's internal index to the vertex group name.
    """
    kd_size = len(vg_to_vertex.keys())
    kd = mathutils.kdtree.KDTree(kd_size)

    idx_to_vggroup = {}

    for idx, (vg, vert) in enumerate(vg_to_vertex.items()):
        kd.insert(vert, idx)
        idx_to_vggroup[idx] = vg

    kd.balance()

    return kd, idx_to_vggroup


def calc_best_candidates(source_obj, dest_obj, weighted=False):
    """Calculates the best vertex group candidate in source for each vertex group in destination.

    Note: This does not mutate anything, only calculates.

    Args:
        source_obj: (bpy_types.Object) Where to read vertex group information from.
        dest_obj: (bpy_types.Object) Which object to have its vertex groups remapped.
        weighted: (bool) Use vertex weights when calculating or not.

    Returns:
        (dict[str, str]) A map of destination object vertex group to source object vertex group.
            If the map is {'3' : '5'}, this means that vertex group '3' should be renamed to '5' in destination.
            If the map is {'つま先.L':'5'}, this means that vertex group 'つま先.L' should be renamed to '5' in destination.

    """
    best = {}

    source_cm = calc_center_of_mass_per_vg(source_obj, weighted)
    source_kd, source_idx_to_vggroup = init_kdtree_vertex_to_cm(source_cm)

    dest_cm = calc_center_of_mass_per_vg(dest_obj, weighted)
    for idx, (vg_group, cm) in enumerate(dest_cm.items()):
        _, idx, _ = source_kd.find(cm)
        best[vg_group] = source_idx_to_vggroup[idx]
    
    return best


def main(source, destination, ref_destination, weighted, merge):
    source_object = bpy.data.objects[source]
    destination_object = bpy.data.objects[destination]

    if not ref_destination:
        ref_destination = destination
    ref_destination_object = bpy.data.objects[ref_destination]
    
    best = calc_best_candidates(source_object, ref_destination_object, weighted=weighted)

    # And then go through the list of vertex groups and rename them
    # In order to reduce name conflicts, we add an "x" in front and then remove it later
    # Blender automatically renames duplicate vertex groups by adding .0001, .0002, etc.
    for vg in destination_object.vertex_groups:
        if vg.name not in best:
            destination_object.vertex_groups.remove(vg)
            continue

        vg.name = f'x{best[vg.name]}'

    for vg in destination_object.vertex_groups:
        vg.name = vg.name[1:]
      
    # Finally, fill in missing spots and sort vertex groups 
    missing_groups = set([f"{vg.name}" for vg in source_object.vertex_groups]) - set([x.name.split(".")[0] for x in destination_object.vertex_groups])
    for missing_group in missing_groups:
        destination_object.vertex_groups.new(name=f"{missing_group}")

    if merge:
        merge_vgs(destination_object)
    
    # I'm not sure if it is possible to sort/add vertex groups without setting the current object and using ops
    bpy.context.view_layer.objects.active = destination_object
    bpy.ops.object.vertex_group_sort()

if __name__ == "__main__":
    main(_SOURCE, _DESTINATION, _REF_DEST, _WEIGHTED, _MERGE)