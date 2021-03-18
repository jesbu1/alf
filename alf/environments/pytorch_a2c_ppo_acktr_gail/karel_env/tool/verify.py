from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import h5py
import os
import sys
import argparse
from tqdm import tqdm
import numpy as np
import torch

from prompt_toolkit import prompt

sys.path.insert(0, '.')
sys.path.insert(0, 'karel_env')

from karel_env import karel
from karel_env.dsl import get_DSL
from karel_env.dsl.dsl_parse import parse
from fetch_mapping import fetch_mapping
from karel_env.tool.syntax_checker import PySyntaxChecker


dsl_tokens = None
prl2dsl_mapping = None
dsl2prl_mapping = None
prl_tokens = None


def prl_to_dsl(program_seq):
    global prl2dsl_mapping, dsl_tokens, prl_tokens

    def func(x):
        assert prl_tokens is not None
        return dsl_tokens.index(prl2dsl_mapping[prl_tokens[x]])
    return list(map(func, program_seq))


def dsl_to_prl(program_seq):
    global dsl2prl_mapping, dsl_tokens, prl_tokens

    def func(x):
        assert dsl_tokens is not None
        if dsl2prl_mapping[dsl_tokens[x]] == '#':
            return '#'
        return prl_tokens.index(dsl2prl_mapping[dsl_tokens[x]])
    return list(map(func, program_seq))


def validate_syntax_checker(syntax_checker, program, use_simplified_dsl):
    if use_simplified_dsl:
        # remove DEF run
        program = dsl_to_prl(program[2:])
        # skip checking a program that can't be represented by simplified DSL
        if '#' in program:
            return True
        initial_state = syntax_checker.get_initial_checker_state2()
    else:
        initial_state = syntax_checker.get_initial_checker_state()

    try:
        sequence_mask = syntax_checker.get_sequence_mask(initial_state, program).squeeze()
    except:
        import pdb
        pdb.set_trace()
    # sequence_mask is zero for next valid token for each input token, so following tensor should be all false
    next_token_validity = sequence_mask[torch.arange(len(program)-1), program[1:]]
    return not next_token_validity.any()


def verify():
    global dsl2prl_mapping, prl2dsl_mapping, dsl_tokens, prl_tokens

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_name', type=str, default='karel_default')
    parser.add_argument('--use_simplified_dsl', action='store_true', help='use simplified DSL or not')
    parser.add_argument('--mapping_file', default=None, help='mapping file for simplified DSL')
    args = parser.parse_args()

    # create syntax checker
    # fetch the tokens
    if args.use_simplified_dsl:
        assert args.mapping_file is not None
        dsl2prl_mapping, prl2dsl_mapping, dsl_tokens, prl_tokens = fetch_mapping(args.mapping_file)
        syntax_checker_tokens = prl_tokens
    else:
        _, _, dsl_tokens, _ = fetch_mapping('mapping_karel2prl.txt')
        syntax_checker_tokens = dsl_tokens

    T2I = {token: i for i, token in enumerate(syntax_checker_tokens)}
    T2I['<pad>'] = len(syntax_checker_tokens)
    syntax_checker_tokens.append('<pad>')

    syntax_checker = PySyntaxChecker(T2I, use_cuda=False,
                                     use_simplified_dsl=args.use_simplified_dsl,
                                     new_tokens=syntax_checker_tokens)

    dir_name = args.dir_name
    data_file = os.path.join(dir_name, 'data.hdf5')
    id_file = os.path.join(dir_name, 'id.txt')

    if not os.path.exists(data_file):
        print("data_file path doesn't exist: {}".format(data_file))
        return
    if not os.path.exists(id_file):
        print("id_file path doesn't exist: {}".format(id_file))
        return

    f = h5py.File(data_file, 'r')
    ids = open(id_file, 'r').read().splitlines()

    dsl = get_DSL(seed=123)
    karel_world = karel.Karel_world(make_error=True)

    for idx in tqdm(ids):
        program = f[idx]['program'][()]
        program_str = dsl.intseq2str(program)
        assert validate_syntax_checker(syntax_checker, program.tolist(), args.use_simplified_dsl), 'next token should be part of valid tokens'
        demos = f[idx]['s_h'][()]
        demo_lens = f[idx]['s_h_len'][()]
        for i in range(demos.shape[0]):
            s = demos[i][0]
            try:
                karel_world.set_new_state(s)
                s_h = dsl.run(karel_world, program_str)
            except RuntimeError:
                assert 0, 'Not supposed to happen'
            else:
                karel_world.clear_history()
                assert np.array_equal(demos[i][:demo_lens[i]], np.array(s_h))


if __name__ == "__main__":
    verify()
