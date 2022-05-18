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
import (
	"unsafe"
)

func Version() (major, minor int, err error) {
	var maj, min C.int
	err = result(C.nvrtcVersion(&maj, &min))
	return int(maj), int(min), err
}

type Program struct {
	c C.nvrtcProgram
}

type Include struct {
	Source string
	Name   string
}

func CreateProgram(source, name string, headers ...Include) (Program, error) {
	var program Program
	csource := C.CString(source)
	defer C.free(unsafe.Pointer(csource))
	cname := C.CString(name)
	defer C.free(unsafe.Pointer(cname))

	if len(headers) == 0 {
		err := result(C.nvrtcCreateProgram(&program.c, csource, cname, 0, nil, nil))
		return program, err
	}

	numHeaders := len(headers)
	cheaders := make([]*C.char, numHeaders)
	cincludeNames := make([]*C.char, numHeaders)
	for i, header := range headers {
		cheaders[i] = C.CString(header.Source)
		cincludeNames[i] = C.CString(header.Name)
	}
	defer func() {
		for i := range headers {
			C.free(unsafe.Pointer(cheaders[i]))
			C.free(unsafe.Pointer(cincludeNames[i]))
		}
	}()

	err := result(C.nvrtcCreateProgram(&program.c, csource, cname,
		C.int(numHeaders), &cheaders[0], &cincludeNames[0]))
	return program, err
}

func (program *Program) Destroy() error {
	err := result(C.nvrtcDestroyProgram(&program.c))
	*program = Program{}
	return err
}

func (program *Program) Compile(options ...string) error {
	if len(options) == 0 {
		return result(C.nvrtcCompileProgram(program.c, 0, nil))
	}

	numOptions := len(options)
	coptions := make([]*C.char, numOptions)
	for i, option := range options {
		coptions[i] = C.CString(option)
	}
	defer func() {
		for i := range options {
			C.free(unsafe.Pointer(coptions[i]))
		}
	}()

	return result(C.nvrtcCompileProgram(program.c, C.int(numOptions), &coptions[0]))
}

func (program *Program) GetPTX() (string, error) {
	var size C.size_t
	err := result(C.nvrtcGetPTXSize(program.c, &size))
	if err != nil {
		return "", err
	}

	data := make([]byte, size+1)
	err = result(C.nvrtcGetPTX(program.c, (*C.char)(unsafe.Pointer(&data[0]))))
	return string(data[:size]), err
}

func (program *Program) GetLog() (string, error) {
	var size C.size_t
	err := result(C.nvrtcGetProgramLogSize(program.c, &size))
	if err != nil {
		return "", err
	}

	data := make([]byte, size+1)
	err = result(C.nvrtcGetProgramLog(program.c, (*C.char)(unsafe.Pointer(&data[0]))))
	return string(data[:size]), err
}

func (program *Program) AddNameExpression(nameExpression string) error {
	cstr := C.CString(nameExpression)
	defer C.free(unsafe.Pointer(cstr))
	return result(C.nvrtcAddNameExpression(program.c, cstr))
}

func (program *Program) GetLoweredName(nameExpression string) (string, error) {
	cstr := C.CString(nameExpression)
	defer C.free(unsafe.Pointer(cstr))

	var loweredName *C.char
	err := result(C.nvrtcGetLoweredName(program.c, cstr, &loweredName))
	if err != nil {
		return "", err
	}
	return C.GoString(loweredName), nil
}
