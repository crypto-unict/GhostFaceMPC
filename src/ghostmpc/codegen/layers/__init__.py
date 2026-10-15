"""Per-operation emitters for the generated MP-SPDZ program.

Every module here turns one manifest operation into source text.  They share
the naming and truncation helpers of :mod:`ghostmpc.codegen.naming` and read,
but never build, the fusion plans of :mod:`ghostmpc.codegen.fusion`.
"""
