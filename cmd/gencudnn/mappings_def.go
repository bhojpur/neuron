package main

// Copyright (c) 2018 Bhojpur Consulting Private Limited, India. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

var fnNameMap map[string]string
var enumMappings map[string]string

// contextual is a list of functions that are contextual (i.e. they require the handle)
var contextual map[string]struct{}

// retVals is a list of functions that have return values.
var retVals map[string]map[int]string

// creations is a list of functions that creates shit. The key is the type
var creations map[string][]string

// setFns is a list of functions that sets types. The key is the type
var setFns map[string][]string

var destructions map[string][]string

// alphaBetas is a list of functions that have alphas, betas, in the parameters
var alphaBetas map[string]map[int]string

// memories is a list of functions that require memory in the parameters
var memories map[string]map[int]string

// methods enumerates the methods
var methods map[string][]string

// orphaned
var orphaned map[string]struct{}

var generated = map[string]struct{}{}
