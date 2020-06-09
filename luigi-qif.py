#!/usr/bin/env python3
import click
import collections
import enum
import fractions
import functools
import json
import math
import os
import shlex
import subprocess
import tempfile
import z3


class Expr(collections.namedtuple('Expr', 'min max value')):
    '''
    An Expr represents a numeric value in the program.
    min: a python int which is a lower bound on the value
    max: a python int which is an upper bound on the value
    value: the Z3 symbolic value of the expression.
           It may be a BitVec or Int sort.
    '''
    def __contains__(self, x):
        return self.min <= x and x <= self.max


class Operator(collections.namedtuple('Operator', 'min max value sext_mode')):
    '''
    A collection of functions, for an operator, which are used to construct Exprs
    min, max, value: take the arguments to the operator, and return the pieces
                     of the Expr
    sext_mode: is one of the _sext_mode* functions below. It explains how to
               sign-extend values so that they have the same number of bits.
    '''


ModelCounter = collections.namedtuple('ModelCounter',
                                      'const sext new_var count')


def _sext_mode_binop(mc, args):
    '''
    Sign extend arguments to a binary operator

    mc: is the model counter object
    args: is a sequence of Exprs

    Return the exprs SignEXTended to the minimum number of bits needed to
    represent both of them.
    '''
    a, b = args
    nbits = max(num_bits(a.min, a.max), num_bits(b.min, b.max))
    return [mc.sext(a.value, nbits), mc.sext(b.value, nbits)]


def _sext_mode_ite(mc, args):
    '''
    Sign extend arguments to the if-then-else operator

    mc: is the model counter object
    args: is a sequence of Exprs

    Return the exprs SignEXTended to the minimum number of bits needed to
    represent both of them.
    '''
    cond, if_true, if_false = args
    nbits = max(num_bits(if_true.min, if_true.max),
                num_bits(if_false.min, if_false.max))
    return [
        cond.value,
        mc.sext(if_true.value, nbits),
        mc.sext(if_false.value, nbits)
    ]


def _bitvec_sext(x, nbits):
    current_bits = x.sort().size()
    assert current_bits <= nbits, '%d <= %d' % (current_bits, nbits)
    if nbits > current_bits:
        return z3.SignExt(nbits - current_bits, x)
    else:
        return x


def _bitvec_const(x, nbits=None):
    return z3.BitVecVal(x, num_bits(x, x) if nbits is None else nbits)


def _bitvec_new_var(name, nbits):
    return z3.BitVec(name, nbits)


def _int_sext(x, nbits):
    return x


def _int_const(x, nbits=None):
    return z3.IntVal(x)


def _int_new_var(name, nbits):
    return z3.Int(name)


def num_bits(vmin, vmax):
    return max(1, vmin.bit_length(), vmax.bit_length()) + 1


def _bsat_count(goal, project_onto):
    '''
    Take as input the z3.Goal object to count, and the list of variables to
    project onto. Return the number of satisfying assignments of the variables.
    '''
    project_onto = list(project_onto)
    solver = z3.Solver()
    solver.add(goal)
    count = 0
    while solver.check() == z3.sat:
        expr = []
        count += 1
        m = solver.model()
        if len(m) == 0: return 0
        if count % 1000 == 0:
            # Progress reporting
            print('intermediate count = %d' % count)
        # Tell Z3 to not return the same model again.
        for var in project_onto:
            expr.append(var != m.get_interp(var))
        solver.add(z3.Or(*expr))
    return count


def convert_to_dimacs(goal, project_onto):
    '''
    Convert a Z3 goal into DIMACS so that ApproxMC can understand it.
    '''
    # Based on https://stackoverflow.com/a/33860849
    bits = set()
    for var in project_onto:
        nbits = var.sort().size()
        # Give a name to each bit of each bitvector
        for bit_index in range(nbits):
            name = '%s_%d' % (var, bit_index)
            bits.add(name)
            bit = z3.Bool(name)
            mask = z3.BitVecVal(1 << bit_index, nbits)
            goal.add(bit == ((var & mask) == mask))
    # z3.With(..., args) provides arguments to the "..." tactic.
    tactic = z3.Then(
        'simplify',
        z3.With(
            z3.Tactic('bit-blast'),
            blast_full=True,
            blast_quant=True,
        ),
        'blast-term-ite',
        z3.With(
            z3.Tactic('propagate-values'),
            blast_eq_value=True,
            blast_distinct=True,
        ),
        z3.With(
            z3.Tactic('simplify'),
            blast_eq_value=True,
        ),
        'blast-term-ite',
        'tseitin-cnf',
        'blast-term-ite',
    )
    expr = tactic(goal)
    assert len(expr) == 1
    expr = expr[0]
    dimacs = expr.dimacs()
    # ind is the Independent Set
    ind = set()
    # Parse the dimacs to determine how Z3 maps the boolean variables to the
    # dimacs variable number
    lines = dimacs.split('\n')
    for line in lines:
        if not line.startswith('c '):
            # It's not a comment.
            continue
        # Otherwise assume that this line maps variables to names
        parts = line.split()
        _, number, var = parts
        if var in bits:
            bits.remove(var)
            ind.add(number)
    # TODO: will this always be true?
    assert len(bits) == 0, repr(bits)
    return 'c ind %s 0\n%s\n' % (' '.join(ind), dimacs)


def _approxmc_count(goal, project_onto, verbose):
    # The last line should look like: [appmc] Number of solutions is: 48 x 2^1
    print("Starting the ApproxMC model counter.")
    print(
        "This may take a while. If you want progress reports, pass the --verbose-approxmc flag"
    )
    input = convert_to_dimacs(goal, project_onto)
    try:
        output = subprocess.run(
            "approxmc --seed=1 --threshold=1000 2>&1 " +
            (" | tee /dev/stderr" if verbose else ""),
            shell=True,
            # approxmc does not use its returncode to report errors.
            check=False,
            stdout=subprocess.PIPE,
            input=input.encode('ascii'),
        ).stdout.decode('ascii').strip()
    except subprocess.CalledProcessError as e:
        print(e.stdout.decode('ascii'))
        raise
    lines = output.split('\n')
    last_line = lines[-1]
    base, pow = last_line.split('Number of solutions is: ')[1].split(' x ')
    base = int(base)
    assert pow.startswith('2^')
    pow = int(pow[len('2^'):])
    return base * (2**pow)


MODEL_COUNTERS = {
    'BSat':
    lambda _: ModelCounter(
        const=_int_const,
        sext=_int_sext,
        count=_bsat_count,
        new_var=_int_new_var,
    ),
    'ApproxMC':
    lambda verbose: ModelCounter(
        const=_bitvec_const,
        sext=_bitvec_sext,
        count=functools.partial(_approxmc_count, verbose=verbose),
        new_var=_bitvec_new_var,
    ),
}


def _operators(mc):
    '''
    Construt a dictionary of operators, given the model counter as an argument
    '''
    return {
        '+':
        Operator(
            min=lambda a, b: a.min + b.min,
            max=lambda a, b: a.max + b.max,
            value=lambda a, b: a + b,
            sext_mode=_sext_mode_binop,
        ),
        '-':
        Operator(
            min=lambda a, b: a.min - b.max,
            max=lambda a, b: a.max - b.min,
            value=lambda a, b: a - b,
            sext_mode=_sext_mode_binop,
        ),
        '*':
        Operator(
            min=lambda a, b: min(
                [a.min * b.min, a.min * b.max, a.max * b.min, a.max * b.max]),
            max=lambda a, b: max(
                [a.min * b.min, a.min * b.max, a.max * b.min, a.max * b.max]),
            value=lambda a, b: a * b,
            sext_mode=_sext_mode_binop,
        ),
        'ITE':
        Operator(
            min=lambda cond, a, b: min(a.min, b.min),
            max=lambda cond, a, b: max(a.max, b.max),
            value=lambda cond, a, b: z3.If((cond != mc.const(
                0,
                cond.sort().size()
                if hasattr(cond.sort(), 'size') else None)), a, b),
            sext_mode=_sext_mode_ite,
        ),
        '<':
        Operator(
            min=lambda *xs: 0,
            max=lambda *xs: 1,
            value=lambda a, b: z3.If(a < b, mc.const(1), mc.const(0)),
            sext_mode=_sext_mode_binop,
        ),
        '<=':
        Operator(
            min=lambda *xs: 0,
            max=lambda *xs: 1,
            value=lambda a, b: z3.If(a <= b, mc.const(1), mc.const(0)),
            sext_mode=_sext_mode_binop,
        ),
        '==':
        Operator(
            min=lambda *xs: 0,
            max=lambda *xs: 1,
            value=lambda a, b: z3.If(a == b, mc.const(1), mc.const(0)),
            sext_mode=_sext_mode_binop,
        ),
        '!=':
        Operator(
            min=lambda *xs: 0,
            max=lambda *xs: 1,
            value=lambda a, b: z3.If(a != b, mc.const(1), mc.const(0)),
            sext_mode=_sext_mode_binop,
        ),
        '>':
        Operator(
            min=lambda *xs: 0,
            max=lambda *xs: 1,
            value=lambda a, b: z3.If(a > b, mc.const(1), mc.const(0)),
            sext_mode=_sext_mode_binop,
        ),
        '>=':
        Operator(
            min=lambda *xs: 0,
            max=lambda *xs: 1,
            value=lambda a, b: z3.If(a >= b, mc.const(1), mc.const(0)),
            sext_mode=_sext_mode_binop,
        ),
    }


def spdz_ingest(mpc, priors):
    '''
    Ingest a SPDZ program given paths to the mpc and priors files
    '''
    input = ''
    with open(os.path.join(os.path.dirname(__file__), 'spdz_ingest.py')) as f:
        input += f.read()
    input += '\n\n'
    with open(mpc) as f:
        input += f.read()
    input += '\n\n_print_output()\n'
    env = os.environ
    # Remove PYTHONHOME which a Nix wrapper sets for python3 purposes
    del env['PYTHONHOME']
    result = json.loads(
        subprocess.run(['python2'],
                       env=env,
                       input=input.encode('ascii'),
                       check=True,
                       stdout=subprocess.PIPE).stdout)
    reveal = result['REVEAL']
    raw_mapping = result['MAPPING']
    with open(priors) as f:
        priors = [
            list(map(int,
                     line.strip().split()))
            for line in f.read().strip().split('\n')
        ]
    return reveal, raw_mapping, priors


def dump_graphviz(dump_circuit, reveal, raw_mapping):
    '''
    Write the circuit graph that Luigi-QIF has created to the given file.
    '''
    with open(dump_circuit, 'w') as f:
        f.write('digraph X {\n')
        f.write('graph [fontname = "monospace"];\n')
        f.write('node [fontname = "monospace"];\n')
        f.write('edge [fontname = "monospace"];\n')
        reachable = set()
        queue = [x for x in reveal if isinstance(x, str)]
        while len(queue) > 0:
            gate = queue.pop()
            if gate in reachable: continue
            reachable.add(gate)
            operator, *args = raw_mapping[gate]
            if operator == 'INPUT':
                f.write('%s [label = "Input %d"];\n' % (gate, args[0]))
            else:
                f.write('%s [label="%s(%s)"];\n' % (gate, operator, ', '.join(
                    '_' if isinstance(x, str) else str(x) for x in args)))
                queue += [x for x in args if isinstance(x, str)]
                for x, label in zip(args, ['cond', 'if_true', 'if_false']
                                    if operator == 'ITE' else list(
                                        None for _ in args)):
                    if not isinstance(x, str):
                        continue
                    f.write('%s -> %s' % (x, gate))
                    if label:
                        f.write(' [label="%s"]' % label)
                    f.write(';\n')
        for x in reveal:
            f.write('%s -> REVEAL;\n' % x)
        f.write('}\n')


# The click library is used for the CLI.
@click.group()
@click.option('--priors',
              type=str,
              required=True,
              help='The new-line separated priors file')
@click.option('--mpc',
              type=str,
              required=True,
              help="The input program source code")
@click.option('--model-counter',
              default='BSat',
              type=click.Choice(['BSat', 'ApproxMC'], case_sensitive=False),
              help='which method to use to compute model counts')
@click.option(
    '--dump-circuit',
    default='',
    help='Dump a GraphViz visualization of the circuit to the given file')
@click.option(
    '--verbose-approxmc',
    is_flag=True,
    default=False,
    type=bool,
    help=
    'Turn on verbose output for ApproxMC, if it is selected as the model counter.',
)
@click.pass_context
def luigi(ctx, priors, mpc, model_counter, verbose_approxmc, dump_circuit):
    # ctx is provided by click
    ctx.ensure_object(dict)
    mc = MODEL_COUNTERS[model_counter](verbose_approxmc)
    ctx.obj['model_counter'] = mc

    _OPERATORS = _operators(mc)

    # STEP ONE: Parse and load the arguments
    reveal, raw_mapping, priors = spdz_ingest(mpc, priors)
    if len(dump_circuit) > 0:
        dump_graphviz(dump_circuit, reveal, raw_mapping)
    # STEP TWO: Convert values to our AST
    goal = z3.Goal()
    exprs = dict()

    def get(x):
        '''
        Turn a handle from the JSON mapping into an Expression.
        If the handle is a constant, make it a constant expression.
        Otherwise, look it up in the map.
        '''
        if isinstance(x, int):
            return Expr(min=x, max=x, value=mc.const(x))
        else:
            return exprs[x]

    ctx.obj['inputs'] = []
    # Rather than recursing using the function call stack, we manually mantain a stack
    stack = [(e, 0) for e in reveal]
    while len(stack) > 0:
        name, partial = stack.pop()
        if isinstance(name, int):
            exprs[name] = get(name)
            continue
        operator, *args = raw_mapping[name]
        if operator == 'INPUT':
            # INPUTs are handled specially from other expressions.
            vmin, vmax = priors[args[0]]
            nbits = num_bits(vmin, vmax)
            var = mc.new_var(name, nbits)
            goal.add(mc.const(vmin, nbits) <= var)
            goal.add(var <= mc.const(vmax, nbits))
            ctx.obj['inputs'].append(name)
            exprs[name] = Expr(min=vmin, max=vmax, value=var)
            continue
        # Partial denotes the number of arguments that have been processed.
        while partial < len(args):
            arg = args[partial]
            # If the argument is either a constant, or has already has a
            # corresponding expression, then we can proceed to the next
            # argument.
            if isinstance(arg, int) or arg in exprs:
                partial += 1
            else:
                # Otherwise, we push ourselves back onto the stack, followed by
                # the argument that we need to evalute.
                stack.append((name, partial))
                stack.append((arg, 0))
                break
        else:
            # The loop has exitied without "break"-ing, so we know that:
            assert partial == len(args)
            # Thus, we can get() each argument to get the corresponding Expr
            operator = _OPERATORS[operator]
            args = [get(arg) for arg in args]
            the_min = operator.min(*args)
            the_max = operator.max(*args)
            args = operator.sext_mode(mc, args)
            value = operator.value(*args)
            new_expr = Expr(min=the_min, max=the_max, value=value)
            assert type(new_expr) is Expr
            exprs[name] = new_expr
    ctx.obj['goal'] = goal
    ctx.obj['exprs'] = exprs
    ctx.obj['reveal'] = reveal


@luigi.command()
@click.pass_context
def static_leakage(ctx):
    mc, goal, exprs, reveal, inputs = [
        ctx.obj[x]
        for x in ('model_counter', 'goal', 'exprs', 'reveal', 'inputs')
    ]
    project_onto = []
    for reveal in reveal:
        expr = exprs[reveal]
        var = mc.new_var('out_%s' % reveal, num_bits(expr.min, expr.max))
        project_onto.append(var)
        goal.add(var == expr.value)
    count = mc.count(goal, project_onto)
    click.echo('Number of possible outputs: %d' % count)
    prior_count = 1
    for input in inputs:
        input = exprs[input]
        prior_count *= input.max - input.min + 1
    click.echo('STATIC LEAKAGE: %f bits out of %f bits total' % (
        math.log2(count),
        math.log2(prior_count),
    ))


@luigi.command()
@click.argument('results', nargs=-1, type=int)
@click.pass_context
def dynamic_leakage(ctx, results):
    mc, goal, exprs, reveal, inputs = [
        ctx.obj[x]
        for x in ('model_counter', 'goal', 'exprs', 'reveal', 'inputs')
    ]
    if len(results) != len(reveal):
        raise click.UsageError(
            'The input program has %d outputs, but only %d concrete outputs were provided.'
            % (len(reveal), len(results)))
    for reveal, result in zip(reveal, results):
        reveal = exprs[reveal]
        nbits = max(num_bits(result, result), num_bits(reveal.min, reveal.max))
        result = mc.const(result, nbits)
        goal.add(mc.sext(reveal.value, nbits) == result)
    prior_count = 1
    for input in inputs:
        input = exprs[input]
        prior_count *= input.max - input.min + 1
    posterior_count = mc.count(
        goal, project_onto=[exprs[input].value for input in inputs])
    if posterior_count == 0:
        raise click.ClickException(
            'Posterior count is 0. The provided output is impossible.')
    posterior_vulnerability = fractions.Fraction(1, posterior_count)
    prior_vulnerability = fractions.Fraction(1, prior_count)
    click.echo('Prior vulnerability: %s' % prior_vulnerability)
    click.echo('Posterior vulnerability: %s' % posterior_vulnerability)
    click.echo('Leakage: %f bits out of %f bits' %
               (math.log2(posterior_vulnerability / prior_vulnerability),
                math.log2(prior_count)))


if __name__ == '__main__':
    luigi(obj={})
