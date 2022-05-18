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
	"io"
	"io/ioutil"
	"log"
	"os"
	"os/exec"
	"os/user"
	"path"
	"path/filepath"
	"runtime"
	"strings"

	engine "github.com/bhojpur/neuron/pkg/tensor/generator"
	stamping "github.com/bhojpur/neuron/pkg/version"
)

var (
	gopath, srcpath string
)

func init() {
	srcpath = "./pkg"
	gopath = os.Getenv("GOPATH")
	log.Printf("Bhojpur Neuron %s", stamping.FullVersion())
	log.Printf("neurgen [src=%s] [GOPATH=%s]", srcpath, gopath)

	// now that Go can have a default GOPATH, this checks that path
	if gopath == "" {
		usr, err := user.Current()
		if err != nil {
			log.Fatal(err)
		}
		gopath = path.Join(usr.HomeDir, "go")
		stat, err := os.Stat(gopath)
		if err != nil {
			log.Fatal(err)
		}
		if !stat.IsDir() {
			log.Fatal("You need to define a $GOPATH")
		}
	}
	engine.TensorPkgLoc = path.Join(srcpath, "/tensor")
	engine.NativePkgLoc = path.Join(srcpath, "/tensor/native")
	engine.ExecLoc = path.Join(srcpath, "/tensor/internal/execution")
	engine.StorageLoc = path.Join(srcpath, "/tensor/internal/storage")
}

func main() {
	pregenerate()

	// storage
	pipeline(engine.StorageLoc, "consts.go", engine.Kinds{engine.AllKinds}, engine.GenerateReflectTypes)
	pipeline(engine.StorageLoc, "getset.go", engine.Kinds{engine.AllKinds}, engine.GenerateHeaderGetSet)
	pipeline(engine.TensorPkgLoc, "array_getset.go", engine.Kinds{engine.AllKinds}, engine.GenerateArrayMethods)

	// execution
	pipeline(engine.ExecLoc, "generic_arith_vv.go", engine.Kinds{engine.AllKinds}, engine.GenerateGenericVecVecArith)
	pipeline(engine.ExecLoc, "generic_arith_mixed.go", engine.Kinds{engine.AllKinds}, engine.GenerateGenericMixedArith)
	// pipeline(engine.ExecLoc, "generic_arith.go", engine.Kinds{engine.AllKinds}, engine.GenerateGenericScalarScalarArith) // generate once and manually edit later
	pipeline(engine.ExecLoc, "generic_cmp_vv.go", engine.Kinds{engine.AllKinds}, engine.GenerateGenericVecVecCmp)
	pipeline(engine.ExecLoc, "generic_cmp_mixed.go", engine.Kinds{engine.AllKinds}, engine.GenerateGenericMixedCmp)
	pipeline(engine.ExecLoc, "generic_minmax.go", engine.Kinds{engine.AllKinds}, engine.GenerateMinMax)
	pipeline(engine.ExecLoc, "generic_map.go", engine.Kinds{engine.AllKinds}, engine.GenerateGenericMap)
	pipeline(engine.ExecLoc, "generic_unary.go", engine.Kinds{engine.AllKinds}, engine.GenerateGenericUncondUnary, engine.GenerateGenericCondUnary, engine.GenerateSpecialGenericUnaries)
	pipeline(engine.ExecLoc, "generic_reduce.go", engine.Kinds{engine.AllKinds}, engine.GenerateGenericReduce)
	pipeline(engine.ExecLoc, "generic_argmethods.go", engine.Kinds{engine.AllKinds}, engine.GenerateGenericArgMethods)
	pipeline(engine.TensorPkgLoc, "generic_utils.go", engine.Kinds{engine.AllKinds}, engine.GenerateUtils)

	// level 1 aggregation
	pipeline(engine.ExecLoc, "eng_arith.go", engine.Kinds{engine.AllKinds}, engine.GenerateEArith)
	pipeline(engine.ExecLoc, "eng_map.go", engine.Kinds{engine.AllKinds}, engine.GenerateEMap)
	pipeline(engine.ExecLoc, "eng_cmp.go", engine.Kinds{engine.AllKinds}, engine.GenerateECmp)
	pipeline(engine.ExecLoc, "eng_minmaxbetween.go", engine.Kinds{engine.AllKinds}, engine.GenerateEMinMaxBetween)
	pipeline(engine.ExecLoc, "eng_reduce.go", engine.Kinds{engine.AllKinds}, engine.GenerateEReduce)
	pipeline(engine.ExecLoc, "eng_unary.go", engine.Kinds{engine.AllKinds}, engine.GenerateUncondEUnary, engine.GenerateCondEUnary, engine.GenerateSpecialEUnaries)
	pipeline(engine.ExecLoc, "reduction_specialization.go", engine.Kinds{engine.AllKinds}, engine.GenerateReductionSpecialization)
	pipeline(engine.ExecLoc, "eng_argmethods.go", engine.Kinds{engine.AllKinds}, engine.GenerateInternalEngArgmethods)

	// level 2 aggregation
	pipeline(engine.TensorPkgLoc, "defaultengine_arith.go", engine.Kinds{engine.AllKinds}, engine.GenerateStdEngArith)
	pipeline(engine.TensorPkgLoc, "defaultengine_cmp.go", engine.Kinds{engine.AllKinds}, engine.GenerateStdEngCmp)
	pipeline(engine.TensorPkgLoc, "defaultengine_unary.go", engine.Kinds{engine.AllKinds}, engine.GenerateStdEngUncondUnary, engine.GenerateStdEngCondUnary)
	pipeline(engine.TensorPkgLoc, "defaultengine_minmax.go", engine.Kinds{engine.AllKinds}, engine.GenerateStdEngMinMax)

	// level 3 aggregation
	pipeline(engine.TensorPkgLoc, "dense_arith.go", engine.Kinds{engine.AllKinds}, engine.GenerateDenseArith)
	pipeline(engine.TensorPkgLoc, "dense_cmp.go", engine.Kinds{engine.AllKinds}, engine.GenerateDenseCmp) // generate once, manually edit later

	// level 4 aggregation
	pipeline(engine.TensorPkgLoc, "api_unary.go", engine.Kinds{engine.AllKinds}, engine.GenerateUncondUnaryAPI, engine.GenerateCondUnaryAPI, engine.GenerateSpecialUnaryAPI)

	// dense methods (old Bhojpur Neuron style)
	pipeline(engine.TensorPkgLoc, "dense_generated.go", engine.Kinds{engine.AllKinds}, engine.GenerateDenseConstructionFns)
	pipeline(engine.TensorPkgLoc, "dense_io.go", engine.Kinds{engine.AllKinds}, engine.GenerateDenseIO)
	pipeline(engine.TensorPkgLoc, "dense_compat.go", engine.Kinds{engine.AllKinds}, engine.GenerateDenseCompat)
	pipeline(engine.TensorPkgLoc, "dense_maskcmp_methods.go", engine.Kinds{engine.AllKinds}, engine.GenerateDenseMaskedMethods)

	// tests
	pipeline(engine.TensorPkgLoc, "test_test.go", engine.Kinds{engine.AllKinds}, engine.GenerateTestUtils)
	pipeline(engine.TensorPkgLoc, "dense_argmethods_test.go", engine.Kinds{engine.AllKinds}, engine.GenerateArgmethodsTests)
	pipeline(engine.TensorPkgLoc, "dense_getset_test.go", engine.Kinds{engine.AllKinds}, engine.GenerateDenseGetSetTests)

	// old-generator style tests
	pipeline(engine.TensorPkgLoc, "dense_reduction_test.go", engine.Kinds{engine.AllKinds}, engine.GenerateDenseReductionTests, engine.GenerateDenseReductionMethodsTests)
	pipeline(engine.TensorPkgLoc, "dense_compat_test.go", engine.Kinds{engine.AllKinds}, engine.GenerateDenseCompatTests)
	pipeline(engine.TensorPkgLoc, "dense_generated_test.go", engine.Kinds{engine.AllKinds}, engine.GenerateDenseConsTests)
	pipeline(engine.TensorPkgLoc, "dense_maskcmp_methods_test.go", engine.Kinds{engine.AllKinds}, engine.GenerateMaskCmpMethodsTests)

	// qc-style tests
	pipeline(engine.TensorPkgLoc, "api_arith_generated_test.go", engine.Kinds{engine.AllKinds}, engine.GenerateAPIArithTests, engine.GenerateAPIArithScalarTests)
	pipeline(engine.TensorPkgLoc, "dense_arith_test.go", engine.Kinds{engine.AllKinds}, engine.GenerateDenseMethodArithTests, engine.GenerateDenseMethodScalarTests)
	pipeline(engine.TensorPkgLoc, "api_unary_generated_test.go", engine.Kinds{engine.AllKinds}, engine.GenerateAPIUnaryTests)
	pipeline(engine.TensorPkgLoc, "api_cmp_generated_test.go", engine.Kinds{engine.AllKinds}, engine.GenerateAPICmpTests, engine.GenerateAPICmpMixedTests)
	pipeline(engine.TensorPkgLoc, "dense_cmp_test.go", engine.Kinds{engine.AllKinds}, engine.GenerateDenseMethodCmpTests, engine.GenerateDenseMethodCmpMixedTests)

	// native iterators
	pipeline(engine.NativePkgLoc, "iterator_native.go", engine.Kinds{engine.AllKinds}, engine.GenerateNativeIterators)
	pipeline(engine.NativePkgLoc, "iterator_native_test.go", engine.Kinds{engine.AllKinds}, engine.GenerateNativeIteratorTests)
	pipeline(engine.NativePkgLoc, "iterator_native2.go", engine.Kinds{engine.AllKinds}, engine.GenerateNativeSelect)
	pipeline(engine.NativePkgLoc, "iterator_native2_test.go", engine.Kinds{engine.AllKinds}, engine.GenerateNativeSelectTests)
}

func pipeline(pkg, filename string, kinds engine.Kinds, fns ...func(io.Writer, engine.Kinds)) {
	fullpath := path.Join(pkg, filename)
	f, err := os.Create(fullpath)
	if err != nil {
		log.Printf("fullpath %q", fullpath)
		log.Fatal(err)
	}
	defer f.Close()
	log.Printf("generating [%s] now", f.Name())
	engine.WritePkgName(f, pkg)

	for _, fn := range fns {
		log.Printf("for kinds [%s]", kinds)
		fn(f, kinds)
	}

	log.Printf("executing [%s]", fullpath)
	// gofmt and goimports this stuff
	cmd := exec.Command("goimports", "-w", fullpath)
	if err = cmd.Run(); err != nil {
		log.Fatalf("Go imports failed with %v for %q", err, fullpath)
	}

	log.Printf("sed assert [%s]", fullpath)
	// account for differences in the postix from the linux sed
	if runtime.GOOS == "darwin" || strings.HasSuffix(runtime.GOOS, "bsd") {
		cmd = exec.Command("sed", "-i", "", `s/github.com\/alecthomas\/assert/github.com\/stretchr\/testify\/assert/g`, fullpath)
	} else {
		cmd = exec.Command("sed", "-E", "-i", `s/github.com\/alecthomas\/assert/github.com\/stretchr\/testify\/assert/g`, fullpath)
	}
	if err = cmd.Run(); err != nil {
		if err.Error() != "exit status 4" { // exit status 4 == not found
			log.Fatalf("sed failed with %v for %q", err.Error(), fullpath)
		}
	}

	log.Printf("formatting [%s]", fullpath)
	cmd = exec.Command("gofmt", "-s", "-w", fullpath)
	if err = cmd.Run(); err != nil {
		log.Fatalf("Gofmt failed for %q", fullpath)
	}
}

// pregenerate cleans up all files that were previously generated.
func pregenerate() error {
	if err := cleanup(engine.StorageLoc); err != nil {
		return err
	}
	if err := cleanup(engine.ExecLoc); err != nil {
		return err
	}
	if err := cleanup(engine.NativePkgLoc); err != nil {
		return err
	}
	return cleanup(engine.TensorPkgLoc)
}

func cleanup(loc string) error {
	pattern := path.Join(loc, "*.go")
	matches, err := filepath.Glob(pattern)
	if err != nil {
		return err
	}
	for _, m := range matches {
		b, err := ioutil.ReadFile(m)
		if err != nil {
			return err
		}
		s := string(b)
		if strings.Contains(s, engine.Genmsg) {
			if err := os.Remove(m); err != nil {
				return err
			}
		}
	}
	return nil
}
