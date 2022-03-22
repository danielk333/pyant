#!/usr/bin/env python

'''
Collection of pre-defined radar beams of different models.

These instances usually correspond to a real physical system.

To register a new radar beam model use the registartor function to pass it the
function that generate the specific radar beam. The function expects a 
signiature of only keyword arguments which are options for the generating
function.

'''


from .beams import beam_of_radar