package nvrtc

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

//#include <nvrtc.h>
import "C"

type nvrtcResult int

func (err nvrtcResult) Error() string  { return err.String() }
func (err nvrtcResult) String() string { return resString[err] }

func result(x C.nvrtcResult) error {
	err := nvrtcResult(x)
	if err == Success {
		return nil
	}
	if err > InternalError {
		return InternalError
	}

	return err
}

const (
	Success                           nvrtcResult = C.NVRTC_SUCCESS
	OutOfMemory                       nvrtcResult = C.NVRTC_ERROR_OUT_OF_MEMORY
	ProgramCreationFailure            nvrtcResult = C.NVRTC_ERROR_PROGRAM_CREATION_FAILURE
	InvalidInput                      nvrtcResult = C.NVRTC_ERROR_INVALID_INPUT
	InvalidProgram                    nvrtcResult = C.NVRTC_ERROR_INVALID_PROGRAM
	InvalidOption                     nvrtcResult = C.NVRTC_ERROR_INVALID_OPTION
	Compilation                       nvrtcResult = C.NVRTC_ERROR_COMPILATION
	BuiltinOperationFailure           nvrtcResult = C.NVRTC_ERROR_BUILTIN_OPERATION_FAILURE
	NoNameExpressionsAfterCompilation nvrtcResult = C.NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION
	NoLoweredNamesBeforeCompilation   nvrtcResult = C.NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION
	NameExpressionNotValid            nvrtcResult = C.NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID
	InternalError                     nvrtcResult = C.NVRTC_ERROR_INTERNAL_ERROR
)

var resString = map[nvrtcResult]string{
	Success:                           "Success",
	OutOfMemory:                       "OutOfMemory",
	ProgramCreationFailure:            "ProgramCreationFailure",
	InvalidInput:                      "InvalidInput",
	InvalidProgram:                    "InvalidProgram",
	InvalidOption:                     "InvalidOption",
	Compilation:                       "Compilation",
	BuiltinOperationFailure:           "BuiltinOperationFailure",
	NoNameExpressionsAfterCompilation: "NoNameExpressionsAfterCompilation",
	NoLoweredNamesBeforeCompilation:   "NoLoweredNamesBeforeCompilation",
	NameExpressionNotValid:            "NameExpressionNotValid",
	InternalError:                     "InternalError",
}
