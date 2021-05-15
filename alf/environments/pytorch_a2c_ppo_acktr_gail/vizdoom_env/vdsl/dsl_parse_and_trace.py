from collections import defaultdict

MONSTER_LIST = ['Demon', 'HellKnight', 'Revenant']

ITEMS_IN_INTEREST = ['MyAmmo']

ACTION_LIST = ['MOVE_FORWARD', 'MOVE_BACKWARD', 'MOVE_LEFT', 'MOVE_RIGHT',
               'TURN_LEFT', 'TURN_RIGHT', 'ATTACK',
               'SELECT_WEAPON1', 'SELECT_WEAPON2', 'SELECT_WEAPON3',
               'SELECT_WEAPON4', 'SELECT_WEAPON5']

DISTANCE_DICT = {
    'doncare_dist': lambda d: True,
    'far': lambda d: d > 400,
    'mid': lambda d: d < 300,
    'close': lambda d: d < 180,
    'very_close': lambda d: d < 135}

HORIZONTAL_DICT = {
    'doncare_horz': lambda l, r, x: True,
    'center': lambda l, r, x: l < x and x < r,
    'slight_left': lambda l, r, x: r < x and x <= r + 10,
    'slight_right': lambda l, r, x: l > x and x >= l - 10,
    'mid_left': lambda l, r, x: r < x and x <= r + 20,
    'mid_right': lambda l, r, x: l > x and x >= l - 20,
    'left': lambda l, r, x: r < x,
    'right': lambda l, r, x: l > x}

CLEAR_DISTANCE_DICT = {
    'far': lambda d: d > 400,
    'mid_far': lambda d: 300 < d and d <= 400,
    'mid': lambda d: 180 < d and d <= 300,
    'close': lambda d: 135 < d and d <= 180,
    'very_close': lambda d: d <= 135}

CLEAR_HORIZONTAL_DICT = {
    'slight_left': lambda l, r, x: r < x and x <= r + 10,
    'slight_right': lambda l, r, x: l > x and x >= l - 10,
    'mid_left': lambda l, r, x: r + 10 < x and x <= r + 20,
    'mid_right': lambda l, r, x: l - 10 > x and x >= l - 20,
    'left': lambda l, r, x: r + 20 < x,
    'right': lambda l, r, x: l - 20 > x}

merge_distance_vocab = list(set(DISTANCE_DICT.keys()).union(
    set(CLEAR_DISTANCE_DICT.keys())))
merge_horizontal_vocab = list(set(HORIZONTAL_DICT.keys()).union(
    set(CLEAR_HORIZONTAL_DICT.keys())))

import types
import functools
def copy_func(f):
    """Based on http://stackoverflow.com/a/6528148/190597 (Glenn Maynard)"""
    g = types.FunctionType(f.__code__, f.__globals__, name=f.__name__,
            argdefs=f.__defaults__,
            closure=f.__closure__)
    g = functools.update_wrapper(g, f)
    g.__kwdefaults__ = f.__kwdefaults__
    return g

def check_and_apply(queue, rule):
    r = rule[0].split()
    l = len(r)
    if len(queue) >= l:
        t = queue[-l:]
        if list(list(zip(*t))[0]) == r:
            new_t = rule[1](list(list(zip(*t))[1]))
            del queue[-l:]
            queue.extend(new_t)
            return True
    return False

rules = []

# world, n, s = fn(world, n)
# world: vizdoom_world
# n: num_call
# s: success
# c: condition [True, False]
MAX_FUNC_CALL = 100


def r_prog_trace(t):
    stmt = t[3]

    def fn(world, n, record_dict, key):
        if n > MAX_FUNC_CALL: return world, n, False
        return stmt(world, n + 1, record_dict, stmt)
    return [('prog', fn)]
rules.append(('DEF run m( stmt m)', r_prog_trace))


def r_stmt_trace(t):
    stmt = t[0]

    def fn(world, n, record_dict, key):
        if n > MAX_FUNC_CALL: return world, n, False
        return stmt(world, n + 1, record_dict, stmt)
    return [('stmt', fn)]
rules.append(('while_stmt', r_stmt_trace))
rules.append(('repeat_stmt', r_stmt_trace))
rules.append(('stmt_stmt', r_stmt_trace))
rules.append(('action', r_stmt_trace))
rules.append(('if_stmt', r_stmt_trace))
rules.append(('ifelse_stmt', r_stmt_trace))


def r_stmt_stmt_trace(t):
    stmt1, stmt2 = t[0], t[1]

    def fn(world, n, record_dict, key):
        if n > MAX_FUNC_CALL: return world, n, False
        world, n, s = stmt1(world, n + 1, record_dict, stmt1)
        if not s: return world, n, s
        if n > MAX_FUNC_CALL: return world, n, False
        return stmt2(world, n, record_dict, stmt2)
    return [('stmt_stmt', fn)]
rules.append(('stmt stmt', r_stmt_stmt_trace))


def r_if_trace(t):
    cond, stmt = t[2], t[5]

    def fn(world, n, record_dict, key):
        if n > MAX_FUNC_CALL: return world, n, False
        world, n, s, c = cond(world, n + 1, record_dict, cond)
        assert len(record_dict[key]) == 1
        record_dict[key][0][1][c] = True
        if not s: return world, n, s
        if c: return stmt(world, n, record_dict, stmt)
        else: return world, n, s
    return [('if_stmt', copy_func(fn))]
rules.append(('IF c( cond c) i( stmt i)', r_if_trace))


def r_ifelse_trace(t):
    cond, stmt1, stmt2 = t[2], t[5], t[9]

    def fn(world, n, record_dict, key):
        if n > MAX_FUNC_CALL: return world, n, False
        world, n, s, c = cond(world, n + 1, record_dict, cond)
        assert len(record_dict[key]) == 1
        record_dict[key][0][1][c] = True
        if not s: return world, n, s
        if c: return stmt1(world, n, record_dict, stmt1)
        else: return stmt2(world, n, record_dict, stmt2)
    return [('ifelse_stmt', copy_func(fn))]
rules.append(('IFELSE c( cond c) i( stmt i) ELSE e( stmt e)', r_ifelse_trace))


def r_while_trace(t):
    cond, stmt = t[2], t[5]

    def fn(world, n, record_dict, key):
        if n > MAX_FUNC_CALL: return world, n, False
        world, n, s, c = cond(world, n, record_dict, cond)
        assert len(record_dict[key]) == 1
        record_dict[key][0][1][c] = True
        if not s: return world, n, s
        while(c):
            world, n, s = stmt(world, n, record_dict, stmt)
            if not s: return world, n, s
            world, n, s, c = cond(world, n, record_dict, cond)
            assert len(record_dict[key]) == 1
            record_dict[key][0][1][c] = True
            if not s: return world, n, s
        return world, n, s
    return [('while_stmt', copy_func(fn))]
rules.append(('WHILE c( cond c) w( stmt w)', r_while_trace))


def r_repeat_trace(t):
    cste, stmt = t[1], t[3]

    def fn(world, n, record_dict, key):
        if n > MAX_FUNC_CALL: return world, n, False
        n += 1
        s = True
        for _ in range(cste()):
            world, n, s = stmt(world, n, record_dict, stmt)
            if not s: return world, n, s
        return world, n, s
    return [('repeat_stmt', fn)]
rules.append(('REPEAT cste r( stmt r)', r_repeat_trace))


def r_cond1_trace(t):
    cond = t[0]

    def fn(world, n, record_dict, key):
        if n > MAX_FUNC_CALL: return world, n, False, False
        return cond(world, n, record_dict, key)
    return [('cond', fn)]
rules.append(('percept', r_cond1_trace))


def r_cond2_trace(t):
    cond = t[2]

    def fn(world, n, record_dict, key):
        if n > MAX_FUNC_CALL: return world, n, False, False
        world, n, s, c = cond(world, n, record_dict, key)
        return world, n, s, not c
    return [('cond', fn)]
rules.append(('not c( cond c)', r_cond2_trace))


def r_percept1_trace(t):
    actor, dist, horz = t[1], t[3], t[4]

    def fn(world, n, record_dict, key):
        if n > MAX_FUNC_CALL: return world, n, False, False
        c = world.exist_actor_in_distance_horizontal(actor(), dist(), horz())
        return world, n, True, c
    return [('percept', fn)]
rules.append(('EXIST actor IN distance horizontal', r_percept1_trace))


def r_percept2_trace(t):
    actor = t[1]

    def fn(world, n, record_dict, key):
        if n > MAX_FUNC_CALL: return world, n, False, False
        c = world.in_target(actor())
        return world, n, True, c
    return [('percept', fn)]
rules.append(('INTARGET actor', r_percept2_trace))


def r_percept3_trace(t):
    actor = t[1]

    def fn(world, n, record_dict, key):
        if n > MAX_FUNC_CALL: return world, n, False, False
        c = world.is_there(actor())
        return world, n, True, c
    return [('percept', fn)]
rules.append(('ISTHERE actor', r_percept3_trace))


def r_actor1_trace(t):
    return [('actor', t[0])]
rules.append(('monster', r_actor1_trace))


def create_r_monster_trace(monster):
    def r_monster(t):
        return [('monster', lambda: monster)]
    return r_monster
for monster in MONSTER_LIST:
    rules.append((monster, create_r_monster_trace(monster)))


def r_actor2_trace(t):
    return [('actor', t[0])]
rules.append(('items', r_actor2_trace))


def create_r_item_trace(item):
    def r_item(t):
        return [('items', lambda: item)]
    return r_item
for item in ITEMS_IN_INTEREST:
    rules.append((item, create_r_item_trace(item)))


def create_r_distance_trace(distance):
    def r_distance(t):
        return [('distance', lambda: distance)]
    return r_distance
for distance in merge_distance_vocab:
    rules.append((distance, create_r_distance_trace(distance)))


def create_r_horizontal_trace(horizontal):
    def r_horizontal(t):
        return [('horizontal', lambda: horizontal)]
    return r_horizontal
for horizontal in merge_horizontal_vocab:
    rules.append((horizontal, create_r_horizontal_trace(horizontal)))


def create_r_slot_trace(slot_number):
    def r_slot(t):
        return [('slot', lambda: slot_number)]
    return r_slot
for slot_number in range(1, 7):
    rules.append(('S={}'.format(slot_number), create_r_slot_trace(slot_number)))


def create_r_action_trace(action):
    def r_action(t):
        def fn(world, n, record_dict, key):
            if n > MAX_FUNC_CALL: world, n, False
            try: world.state_transition(action)
            except: return world, n, False
            else: return world, n, True
        return [('action', fn)]
    return r_action
for action in ACTION_LIST:
    rules.append((action, create_r_action_trace(action)))


def create_r_cste_trace(number):
    def r_cste(t):
        return [('cste', lambda: number)]
    return r_cste
for i in range(20):
    rules.append(('R={}'.format(i), create_r_cste_trace(i)))


def parse_and_trace(program, environment='vizdoom'):
    record_dict = defaultdict(list)
    p_tokens = program.split()[::-1]
    p_current_lexpos = -1
    queue = []
    applied = False
    while len(p_tokens) > 0 or len(queue) != 1:
        if applied: applied = False
        else:
            queue.append((p_tokens.pop(), None))
            p_current_lexpos += 1
        for rule in rules:
            applied = check_and_apply(queue, rule)
            if applied:
                if 'WHILE' in rule[0] or 'IF' in rule[0]:
                    #print("D:", rule, queue[-1][1], p_current_lexpos)
                    record_dict[queue[-1][1]].append([p_current_lexpos, {True: False, False: False}])
                break
        if not applied and len(p_tokens) == 0:  # error parsing
            return None, False, {}
    return queue[0][1], True, record_dict
