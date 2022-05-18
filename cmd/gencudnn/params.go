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

import (
	bg "github.com/bhojpur/neuron/pkg/bindgen"
	"modernc.org/cc"
)

func isInput(fnName string, p bg.Parameter) bool {
	return inList(p.Name(), inputParams[fnName])
}

func isOutput(fnName string, p bg.Parameter) bool {
	return inList(p.Name(), outputParams[fnName])
}

func isIO(fnName string, p bg.Parameter) bool {
	return inList(p.Name(), ioParams[fnName])
}

func isAlphaBeta(fnName string, p bg.Parameter) bool {
	locs := alphaBetas[fnName]
	for _, v := range locs {
		if v == p.Name() {
			return true
		}
	}
	return false
}

// functions for convertibility
func isOutputPtrOfPrim(fnName string, p bg.Parameter) bool {
	if !isOutput(fnName, p) && !isIO(fnName, p) {
		return false
	}
	if !p.IsPointer() {
		return false
	}
	return isBuiltin(depointerize(nameOfType(p.Type())))
}

func isEnumOutput(fnName string, p bg.Parameter) bool {
	if !isOutput(fnName, p) {
		return false
	}
	if !p.IsPointer() {
		return false
	}
	cType := nameOfType(p.Type())
	_, ok := enumMappings[cType]
	return ok
}

func cParam2GoParam(p bg.Parameter) (retVal Param) {
	retVal.Name = safeParamName(p.Name())
	cTypeName := nameOfType(p.Type())
	gTypeName := goNameOf(p.Type())
	isPtr, isBuiltin := isPointerOfBuiltin(cTypeName)

	switch {
	case gTypeName == "" && isPtr && isBuiltin:
		retVal.Type = goNameOfStr(depointerize(cTypeName))
	case gTypeName != "":
		retVal.Type = gTypeName
	case gTypeName == "" && !isBuiltin:
		retVal.Type = "TODO"
	}
	return
}

func ctype2gotype2ctype(t cc.Type) string {
	cName := nameOfType(t)
	goName := goNameOfStr(depointerize(cName))
	return go2cBuiltins[goName]
}
