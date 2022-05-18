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
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"path"
)

var pkgloc string
var apiFile string
var ctxFile string
var resultFile string

func init() {
	gopath := os.Getenv("GOPATH")
	pkgloc = path.Join(gopath, "src/github.com/bhojpur/neuron/pkg/drivers/cu")
	apiFile = path.Join(pkgloc, "api.go")
	ctxFile = path.Join(pkgloc, "ctx_api.go")
	resultFile = path.Join(pkgloc, "result.go")
}

func generateAPIFile(gss []*GoSignature) {
	var original []byte
	if _, err := os.Stat(apiFile); err == nil {
		if original, err = ioutil.ReadFile(apiFile); err != nil {
			panic(err)
		}
	}

	f, err := os.Create(apiFile)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	defer func(original []byte, f *os.File) {
		if r := recover(); r != nil {
			f.Truncate(0)
			f.Seek(0, 0)
			f.Write(original)
			log.Printf("NO CHANGES MADE TO %v. Generating API errored with %v", apiFile, r)
			var errfmt string
			for k := range errs {
				errfmt += "\n"
				errfmt += k
			}
			log.Printf("Errors:%v", errfmt)

		}
	}(original, f)

	f.WriteString(header)
	generateAPI(f, gss)
}

func generateContextFile(gss []*GoSignature) {
	var original []byte
	if _, err := os.Stat(ctxFile); err == nil {
		if original, err = ioutil.ReadFile(ctxFile); err != nil {
			panic(err)
		}
	}

	g, err := os.Create(ctxFile)
	if err != nil {
		panic(err)
	}
	defer g.Close()
	defer func(original []byte, f *os.File) {
		if r := recover(); r != nil {
			f.Truncate(0)
			f.Seek(0, 0)
			f.Write(original)
			log.Printf("NO CHANGES MADE TO %v. Generating Context errored with\n%v", ctxFile, r)
		}
	}(original, g)
	g.WriteString(header)
	generateContextAPI(g, gss)
}

func generateResultFile() {
	g, err := os.Create(resultFile)
	if err != nil {
		panic(err)
	}
	defer g.Close()

	g.WriteString(resultHeader)
	generateResultEnums(g)
}

func main() {
	// input := strings.NewReader(src)
	// sigs := Parse(input)
	sigs := Parse()
	sigs = filterCSigs(sigs)
	//	fmt.Printf("Sigs\n%v", sigs)

	var gss []*GoSignature

	for _, sig := range sigs {
		gs := sig.GoSig()
		gss = append(gss, gs)
	}

	//generateResultFile()
	generateAPIFile(gss)
	//generateContextFile(gss)

	var err error
	files := []string{
		apiFile,
		ctxFile,
		resultFile,
	}

	for _, filename := range files {
		cmd := exec.Command("goimports", "-w", filename)
		if err = cmd.Run(); err != nil {
			log.Printf("Go imports failed with %v for %q", err, filename)
		}
	}

}
