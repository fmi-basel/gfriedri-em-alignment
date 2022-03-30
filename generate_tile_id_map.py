import os
import glob
import json
import itertools
import warnings
import numpy as np

def parse_adoc(lines):
    # converts an adoc-format string list into a dictionary
    output = {}
    for line in lines:
        entry = line.split()
        if entry: output.update({entry[0]: entry[2:]})

    return output


def load_tile_json(line):
    instr = line[line.find('{'):].replace("'",'"')
    tile = json.loads(instr)
    return tile

def tile_spec_from_SBEMtile(line, pxs):
    tile = load_tile_json(line)
    tile_spec = {}
    tile_spec['tile_id'] = int(tile['tileid'].split('.')[1])
    tile_spec['tile_file'] = os.path.realpath(tile['filename'])
    tile_spec['x'] = float(tile['glob_x']) // pxs
    tile_spec['y'] = float(tile['glob_y']) // pxs
    tile_spec['z'] = tile['slice_counter']
    return tile_spec


def tile_spec_from_sbemimage (imgdir):
    print('Loading from {}'.format(imgdir))
    os.chdir(imgdir)
    if not os.path.exists('meta'): raise  FileNotFoundError('Change to proper SBEMimage data directory!')

    mfile0 = os.path.join('meta','logs','metadata_')

    mfiles = glob.glob(mfile0+'*')
    mfiles = sorted(mfiles)

    tile_spec_list=[]
    allspecs = []
    curr_res_xy = -1
    stack_idx = 0
    stackname='stack'

    for mfile in mfiles:
        print(mfile)
        acq_suffix = mfile[mfile.rfind('_'):]
        with open(mfile) as mdf: mdl = mdf.read().splitlines()

        conffile = os.path.join('meta','logs','config'+acq_suffix)
        with open(conffile) as cf: cl = cf.read().splitlines()
        config = parse_adoc(cl[:cl.index('[overviews]')])
        pxs = float(config['pixel_size'][0].strip('[],'))#/1000  # in um
        z_thick = float(config['slice_thickness'][0])#/1000  # in um
        resolution_xy = (pxs,pxs)

        if not curr_res_xy == -1:
            if not resolution_xy==curr_res_xy:
                stack_idx += 1
                allspecs.append((stackname,tile_spec_list,curr_res_xy))
                stackname += '_' + '%02d' %stack_idx
                tile_spec_list=[]

        curr_res_xy = resolution_xy

        for line in mdl:
            if line.startswith('TILE: '):

                tile_spec  = tile_spec_from_SBEMtile(line, pxs)

                if not os.path.exists(tile_spec['tile_file']):
                    fnf_error = 'Warning: File '+tile_spec['tile_file']+' does not exist'
                    # print(fnf_error)
                tile_spec_list.append(tile_spec)


    allspecs.append((stackname,tile_spec_list,resolution_xy))

    # only use the 1st stack, as for now the resolution parameters do not change
    # during acquisition
    if len(allspecs) > 1:
        warnings.warn('Acquisition parameters changed. Only returning the 1st stack!')
    return allspecs[0][1]

def combine_tile_spec_from_sbem_runs(imgdir_list):
    tile_spec_dlist = [tile_spec_from_sbemimage(imgdir) for imgdir in imgdir_list]
    section_range_list = [get_section_range(tsl) for tsl in tile_spec_dlist]
    tile_spec_list = list(itertools.chain.from_iterable(tile_spec_dlist))
    return tile_spec_list, section_range_list


def generate_tile_id_map(tile_spec_list):
    tile_to_coord = {}
    for ts in tile_spec_list:
        tile_to_coord[ts['tile_id']] = (ts['x'], ts['y'])

    xx, yy = set(), set()
    for t in tile_to_coord.values():
        xx.add(t[0])
        yy.add(t[1])

    xx = list(sorted(xx))
    yy = list(sorted(yy))

    tiles = np.zeros((len(yy), len(xx)), dtype=int) - 1
    for t, (x, y) in tile_to_coord.items():
        tiles[yy.index(y), xx.index(x)] = t
    return tiles


def get_section_range(tile_spec_list):
    zlist = [ts['z'] for ts in tile_spec_list]
    return min(zlist), max(zlist)


def json_dump(data, file_name):
    with open(file_name, "w") as fp:
        json.dump(data, fp)

def get_imgdir(data_dir, stack_name, run_num):
    return os.path.join(data_dir, '{}_run{:02d}'.format(stack_name, run_num))


def save_tile_id_map(outdir, tile_id_map, imgdir_list, section_range_list):
    np.save(os.path.join(outdir, 'tile_id_map.npy'), tile_id_map)
    section_range = dict(imgdir_list=imgdir_list,
                         section_range_list=section_range_list)
    json_dump(section_range, os.path.join(outdir, 'section_range.json'))


def validate_section(tile_spec_list):
    zlist = np.array([ts['z'] for ts in tile_spec_list])
    zdiff = np.diff(zlist)
    increase_z = zdiff >= 0
    no_missing_slice = zdiff <= 1
    if not increase_z.all():
        print('Image list in valid! Image list should be sorted increasingly according to z.')
    if not no_missing_slice.all():
        miss_idx = zlist[:-1][~no_missing_slice]
        print('Image list in valid! Missing slices!')
    else:
        miss_idx = [];
    valid = increase_z.all() and no_missing_slice.all()
    return valid, miss_idx


if __name__ == '__main__':
    data_dir = 'some_data_dir'
    stack_name = 'example_stack'
    outdir = os.path.join('some_out_dir',
                          stack_name)
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    # run_range = range(1, 12)
    run_range = [4]
    imgdir_list = [get_imgdir(data_dir, stack_name, r) for r in run_range]
    tile_spec_list, section_range_list = combine_tile_spec_from_sbem_runs(imgdir_list)
    validate_section(tile_spec_list)
    tile_id_map = generate_tile_id_map(tile_spec_list)
    save_tile_id_map(outdir, tile_id_map, imgdir_list, section_range_list)
