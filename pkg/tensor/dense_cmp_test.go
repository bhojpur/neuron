// Code generated by Bhojpur Neuron. DO NOT EDIT.

package tensor

import (
	"reflect"
	"testing"
	"testing/quick"
)

func TestDense_Gt(t *testing.T) {
	transFn := func(q *Dense) bool {
		we, _ := willerr(q, ordTypes, nil)
		_, ok := q.Engine().(Gter)
		we = we || !ok

		r := newRand()
		a := q.Clone().(*Dense)
		b := q.Clone().(*Dense)
		c := q.Clone().(*Dense)

		bv, _ := quick.Value(b.Dtype().Type, r)
		cv, _ := quick.Value(c.Dtype().Type, r)
		b.Memset(bv.Interface())
		c.Memset(cv.Interface())

		axb, err := a.Gt(b)
		if err, retEarly := qcErrCheck(t, "Gt - a∙b", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		bxc, err := b.Gt(c)
		if err, retEarly := qcErrCheck(t, "Gt - b∙c", b, c, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		axc, err := a.Gt(c)
		if err, retEarly := qcErrCheck(t, "Gt - a∙c", a, c, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		ab := axb.Bools()
		bc := bxc.Bools()
		ac := axc.Bools()
		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}

		return true
	}
	if err := quick.Check(transFn, &quick.Config{Rand: newRand(), MaxCount: quickchecks}); err != nil {
		t.Errorf("Transitivity test for Gt failed: %v", err)
	}

}
func TestDense_Gte(t *testing.T) {
	transFn := func(q *Dense) bool {
		we, _ := willerr(q, ordTypes, nil)
		_, ok := q.Engine().(Gteer)
		we = we || !ok

		r := newRand()
		a := q.Clone().(*Dense)
		b := q.Clone().(*Dense)
		c := q.Clone().(*Dense)

		bv, _ := quick.Value(b.Dtype().Type, r)
		cv, _ := quick.Value(c.Dtype().Type, r)
		b.Memset(bv.Interface())
		c.Memset(cv.Interface())

		axb, err := a.Gte(b)
		if err, retEarly := qcErrCheck(t, "Gte - a∙b", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		bxc, err := b.Gte(c)
		if err, retEarly := qcErrCheck(t, "Gte - b∙c", b, c, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		axc, err := a.Gte(c)
		if err, retEarly := qcErrCheck(t, "Gte - a∙c", a, c, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		ab := axb.Bools()
		bc := bxc.Bools()
		ac := axc.Bools()
		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}

		return true
	}
	if err := quick.Check(transFn, &quick.Config{Rand: newRand(), MaxCount: quickchecks}); err != nil {
		t.Errorf("Transitivity test for Gte failed: %v", err)
	}

}
func TestDense_Lt(t *testing.T) {
	transFn := func(q *Dense) bool {
		we, _ := willerr(q, ordTypes, nil)
		_, ok := q.Engine().(Lter)
		we = we || !ok

		r := newRand()
		a := q.Clone().(*Dense)
		b := q.Clone().(*Dense)
		c := q.Clone().(*Dense)

		bv, _ := quick.Value(b.Dtype().Type, r)
		cv, _ := quick.Value(c.Dtype().Type, r)
		b.Memset(bv.Interface())
		c.Memset(cv.Interface())

		axb, err := a.Lt(b)
		if err, retEarly := qcErrCheck(t, "Lt - a∙b", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		bxc, err := b.Lt(c)
		if err, retEarly := qcErrCheck(t, "Lt - b∙c", b, c, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		axc, err := a.Lt(c)
		if err, retEarly := qcErrCheck(t, "Lt - a∙c", a, c, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		ab := axb.Bools()
		bc := bxc.Bools()
		ac := axc.Bools()
		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}

		return true
	}
	if err := quick.Check(transFn, &quick.Config{Rand: newRand(), MaxCount: quickchecks}); err != nil {
		t.Errorf("Transitivity test for Lt failed: %v", err)
	}

}
func TestDense_Lte(t *testing.T) {
	transFn := func(q *Dense) bool {
		we, _ := willerr(q, ordTypes, nil)
		_, ok := q.Engine().(Lteer)
		we = we || !ok

		r := newRand()
		a := q.Clone().(*Dense)
		b := q.Clone().(*Dense)
		c := q.Clone().(*Dense)

		bv, _ := quick.Value(b.Dtype().Type, r)
		cv, _ := quick.Value(c.Dtype().Type, r)
		b.Memset(bv.Interface())
		c.Memset(cv.Interface())

		axb, err := a.Lte(b)
		if err, retEarly := qcErrCheck(t, "Lte - a∙b", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		bxc, err := b.Lte(c)
		if err, retEarly := qcErrCheck(t, "Lte - b∙c", b, c, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		axc, err := a.Lte(c)
		if err, retEarly := qcErrCheck(t, "Lte - a∙c", a, c, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		ab := axb.Bools()
		bc := bxc.Bools()
		ac := axc.Bools()
		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}

		return true
	}
	if err := quick.Check(transFn, &quick.Config{Rand: newRand(), MaxCount: quickchecks}); err != nil {
		t.Errorf("Transitivity test for Lte failed: %v", err)
	}

}
func TestDense_ElEq(t *testing.T) {
	transFn := func(q *Dense) bool {
		we, _ := willerr(q, eqTypes, nil)
		_, ok := q.Engine().(ElEqer)
		we = we || !ok

		r := newRand()
		a := q.Clone().(*Dense)
		b := q.Clone().(*Dense)
		c := q.Clone().(*Dense)

		bv, _ := quick.Value(b.Dtype().Type, r)
		cv, _ := quick.Value(c.Dtype().Type, r)
		b.Memset(bv.Interface())
		c.Memset(cv.Interface())

		axb, err := a.ElEq(b)
		if err, retEarly := qcErrCheck(t, "ElEq - a∙b", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		bxc, err := b.ElEq(c)
		if err, retEarly := qcErrCheck(t, "ElEq - b∙c", b, c, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		axc, err := a.ElEq(c)
		if err, retEarly := qcErrCheck(t, "ElEq - a∙c", a, c, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		ab := axb.Bools()
		bc := bxc.Bools()
		ac := axc.Bools()
		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}

		return true
	}
	if err := quick.Check(transFn, &quick.Config{Rand: newRand(), MaxCount: quickchecks}); err != nil {
		t.Errorf("Transitivity test for ElEq failed: %v", err)
	}

	symFn := func(q *Dense) bool {
		we, _ := willerr(q, eqTypes, nil)
		_, ok := q.Engine().(ElEqer)
		we = we || !ok

		r := newRand()
		a := q.Clone().(*Dense)
		b := q.Clone().(*Dense)

		bv, _ := quick.Value(b.Dtype().Type, r)
		b.Memset(bv.Interface())

		axb, err := a.ElEq(b)
		if err, retEarly := qcErrCheck(t, "ElEq - a∙b", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		bxa, err := b.ElEq(a)
		if err, retEarly := qcErrCheck(t, "ElEq - b∙a", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}
		return reflect.DeepEqual(axb.Data(), bxa.Data())

	}
	if err := quick.Check(symFn, &quick.Config{Rand: newRand(), MaxCount: quickchecks}); err != nil {
		t.Errorf("Transitivity test for ElEq failed: %v", err)
	}
}
func TestDense_ElNe(t *testing.T) {
	symFn := func(q *Dense) bool {
		we, _ := willerr(q, eqTypes, nil)
		_, ok := q.Engine().(ElEqer)
		we = we || !ok

		r := newRand()
		a := q.Clone().(*Dense)
		b := q.Clone().(*Dense)

		bv, _ := quick.Value(b.Dtype().Type, r)
		b.Memset(bv.Interface())

		axb, err := a.ElNe(b)
		if err, retEarly := qcErrCheck(t, "ElNe - a∙b", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		bxa, err := b.ElNe(a)
		if err, retEarly := qcErrCheck(t, "ElNe - b∙a", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}
		return reflect.DeepEqual(axb.Data(), bxa.Data())

	}
	if err := quick.Check(symFn, &quick.Config{Rand: newRand(), MaxCount: quickchecks}); err != nil {
		t.Errorf("Transitivity test for ElNe failed: %v", err)
	}
}
func TestDense_Gt_assame(t *testing.T) {
	transFn := func(q *Dense) bool {
		we, _ := willerr(q, nonComplexNumberTypes, nil)
		_, ok := q.Engine().(Gter)
		we = we || !ok

		if err := typeclassCheck(q.Dtype(), nonComplexNumberTypes); err != nil {
			return true // we exit early if the generated type is not something we can handle
		}
		r := newRand()
		a := q.Clone().(*Dense)
		b := q.Clone().(*Dense)
		c := q.Clone().(*Dense)

		bv, _ := quick.Value(b.Dtype().Type, r)
		cv, _ := quick.Value(c.Dtype().Type, r)
		b.Memset(bv.Interface())
		c.Memset(cv.Interface())

		axb, err := a.Gt(b, AsSameType())
		if err, retEarly := qcErrCheck(t, "Gt - a∙b", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		bxc, err := b.Gt(c, AsSameType())
		if err, retEarly := qcErrCheck(t, "Gt - b∙c", b, c, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		axc, err := a.Gt(c, AsSameType())
		if err, retEarly := qcErrCheck(t, "Gt - a∙c", a, c, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		if !threewayEq(axb.Data(), bxc.Data(), axc.Data()) {
			t.Errorf("a: %-v", a)
			t.Errorf("b: %-v", b)
			t.Errorf("c: %-v", c)
			t.Errorf("axb.Data() %v", axb.Data())
			t.Errorf("bxc.Data() %v", bxc.Data())
			t.Errorf("axc.Data() %v", axc.Data())
			return false
		}

		return true
	}
	if err := quick.Check(transFn, &quick.Config{Rand: newRand(), MaxCount: quickchecks}); err != nil {
		t.Errorf("Transitivity test for Gt failed: %v", err)
	}

}
func TestDense_Gte_assame(t *testing.T) {
	transFn := func(q *Dense) bool {
		we, _ := willerr(q, nonComplexNumberTypes, nil)
		_, ok := q.Engine().(Gteer)
		we = we || !ok

		if err := typeclassCheck(q.Dtype(), nonComplexNumberTypes); err != nil {
			return true // we exit early if the generated type is not something we can handle
		}
		r := newRand()
		a := q.Clone().(*Dense)
		b := q.Clone().(*Dense)
		c := q.Clone().(*Dense)

		bv, _ := quick.Value(b.Dtype().Type, r)
		cv, _ := quick.Value(c.Dtype().Type, r)
		b.Memset(bv.Interface())
		c.Memset(cv.Interface())

		axb, err := a.Gte(b, AsSameType())
		if err, retEarly := qcErrCheck(t, "Gte - a∙b", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		bxc, err := b.Gte(c, AsSameType())
		if err, retEarly := qcErrCheck(t, "Gte - b∙c", b, c, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		axc, err := a.Gte(c, AsSameType())
		if err, retEarly := qcErrCheck(t, "Gte - a∙c", a, c, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		if !threewayEq(axb.Data(), bxc.Data(), axc.Data()) {
			t.Errorf("a: %-v", a)
			t.Errorf("b: %-v", b)
			t.Errorf("c: %-v", c)
			t.Errorf("axb.Data() %v", axb.Data())
			t.Errorf("bxc.Data() %v", bxc.Data())
			t.Errorf("axc.Data() %v", axc.Data())
			return false
		}

		return true
	}
	if err := quick.Check(transFn, &quick.Config{Rand: newRand(), MaxCount: quickchecks}); err != nil {
		t.Errorf("Transitivity test for Gte failed: %v", err)
	}

}
func TestDense_Lt_assame(t *testing.T) {
	transFn := func(q *Dense) bool {
		we, _ := willerr(q, nonComplexNumberTypes, nil)
		_, ok := q.Engine().(Lter)
		we = we || !ok

		if err := typeclassCheck(q.Dtype(), nonComplexNumberTypes); err != nil {
			return true // we exit early if the generated type is not something we can handle
		}
		r := newRand()
		a := q.Clone().(*Dense)
		b := q.Clone().(*Dense)
		c := q.Clone().(*Dense)

		bv, _ := quick.Value(b.Dtype().Type, r)
		cv, _ := quick.Value(c.Dtype().Type, r)
		b.Memset(bv.Interface())
		c.Memset(cv.Interface())

		axb, err := a.Lt(b, AsSameType())
		if err, retEarly := qcErrCheck(t, "Lt - a∙b", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		bxc, err := b.Lt(c, AsSameType())
		if err, retEarly := qcErrCheck(t, "Lt - b∙c", b, c, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		axc, err := a.Lt(c, AsSameType())
		if err, retEarly := qcErrCheck(t, "Lt - a∙c", a, c, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		if !threewayEq(axb.Data(), bxc.Data(), axc.Data()) {
			t.Errorf("a: %-v", a)
			t.Errorf("b: %-v", b)
			t.Errorf("c: %-v", c)
			t.Errorf("axb.Data() %v", axb.Data())
			t.Errorf("bxc.Data() %v", bxc.Data())
			t.Errorf("axc.Data() %v", axc.Data())
			return false
		}

		return true
	}
	if err := quick.Check(transFn, &quick.Config{Rand: newRand(), MaxCount: quickchecks}); err != nil {
		t.Errorf("Transitivity test for Lt failed: %v", err)
	}

}
func TestDense_Lte_assame(t *testing.T) {
	transFn := func(q *Dense) bool {
		we, _ := willerr(q, nonComplexNumberTypes, nil)
		_, ok := q.Engine().(Lteer)
		we = we || !ok

		if err := typeclassCheck(q.Dtype(), nonComplexNumberTypes); err != nil {
			return true // we exit early if the generated type is not something we can handle
		}
		r := newRand()
		a := q.Clone().(*Dense)
		b := q.Clone().(*Dense)
		c := q.Clone().(*Dense)

		bv, _ := quick.Value(b.Dtype().Type, r)
		cv, _ := quick.Value(c.Dtype().Type, r)
		b.Memset(bv.Interface())
		c.Memset(cv.Interface())

		axb, err := a.Lte(b, AsSameType())
		if err, retEarly := qcErrCheck(t, "Lte - a∙b", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		bxc, err := b.Lte(c, AsSameType())
		if err, retEarly := qcErrCheck(t, "Lte - b∙c", b, c, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		axc, err := a.Lte(c, AsSameType())
		if err, retEarly := qcErrCheck(t, "Lte - a∙c", a, c, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		if !threewayEq(axb.Data(), bxc.Data(), axc.Data()) {
			t.Errorf("a: %-v", a)
			t.Errorf("b: %-v", b)
			t.Errorf("c: %-v", c)
			t.Errorf("axb.Data() %v", axb.Data())
			t.Errorf("bxc.Data() %v", bxc.Data())
			t.Errorf("axc.Data() %v", axc.Data())
			return false
		}

		return true
	}
	if err := quick.Check(transFn, &quick.Config{Rand: newRand(), MaxCount: quickchecks}); err != nil {
		t.Errorf("Transitivity test for Lte failed: %v", err)
	}

}
func TestDense_ElEq_assame(t *testing.T) {
	transFn := func(q *Dense) bool {
		we, _ := willerr(q, nonComplexNumberTypes, nil)
		_, ok := q.Engine().(ElEqer)
		we = we || !ok

		if err := typeclassCheck(q.Dtype(), nonComplexNumberTypes); err != nil {
			return true // we exit early if the generated type is not something we can handle
		}
		r := newRand()
		a := q.Clone().(*Dense)
		b := q.Clone().(*Dense)
		c := q.Clone().(*Dense)

		bv, _ := quick.Value(b.Dtype().Type, r)
		cv, _ := quick.Value(c.Dtype().Type, r)
		b.Memset(bv.Interface())
		c.Memset(cv.Interface())

		axb, err := a.ElEq(b, AsSameType())
		if err, retEarly := qcErrCheck(t, "ElEq - a∙b", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		bxc, err := b.ElEq(c, AsSameType())
		if err, retEarly := qcErrCheck(t, "ElEq - b∙c", b, c, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		axc, err := a.ElEq(c, AsSameType())
		if err, retEarly := qcErrCheck(t, "ElEq - a∙c", a, c, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		if !threewayEq(axb.Data(), bxc.Data(), axc.Data()) {
			t.Errorf("a: %-v", a)
			t.Errorf("b: %-v", b)
			t.Errorf("c: %-v", c)
			t.Errorf("axb.Data() %v", axb.Data())
			t.Errorf("bxc.Data() %v", bxc.Data())
			t.Errorf("axc.Data() %v", axc.Data())
			return false
		}

		return true
	}
	if err := quick.Check(transFn, &quick.Config{Rand: newRand(), MaxCount: quickchecks}); err != nil {
		t.Errorf("Transitivity test for ElEq failed: %v", err)
	}

	symFn := func(q *Dense) bool {
		we, _ := willerr(q, nonComplexNumberTypes, nil)
		_, ok := q.Engine().(ElEqer)
		we = we || !ok

		if err := typeclassCheck(q.Dtype(), nonComplexNumberTypes); err != nil {
			return true // we exit early if the generated type is not something we can handle
		}
		r := newRand()
		a := q.Clone().(*Dense)
		b := q.Clone().(*Dense)

		bv, _ := quick.Value(b.Dtype().Type, r)
		b.Memset(bv.Interface())

		axb, err := a.ElEq(b, AsSameType())
		if err, retEarly := qcErrCheck(t, "ElEq - a∙b", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		bxa, err := b.ElEq(a, AsSameType())
		if err, retEarly := qcErrCheck(t, "ElEq - b∙a", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}
		return reflect.DeepEqual(axb.Data(), bxa.Data())

	}
	if err := quick.Check(symFn, &quick.Config{Rand: newRand(), MaxCount: quickchecks}); err != nil {
		t.Errorf("Transitivity test for ElEq failed: %v", err)
	}
}
func TestDense_ElNe_assame(t *testing.T) {
	symFn := func(q *Dense) bool {
		we, _ := willerr(q, nonComplexNumberTypes, nil)
		_, ok := q.Engine().(ElEqer)
		we = we || !ok

		if err := typeclassCheck(q.Dtype(), nonComplexNumberTypes); err != nil {
			return true // we exit early if the generated type is not something we can handle
		}
		r := newRand()
		a := q.Clone().(*Dense)
		b := q.Clone().(*Dense)

		bv, _ := quick.Value(b.Dtype().Type, r)
		b.Memset(bv.Interface())

		axb, err := a.ElNe(b, AsSameType())
		if err, retEarly := qcErrCheck(t, "ElNe - a∙b", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		bxa, err := b.ElNe(a, AsSameType())
		if err, retEarly := qcErrCheck(t, "ElNe - b∙a", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}
		return reflect.DeepEqual(axb.Data(), bxa.Data())

	}
	if err := quick.Check(symFn, &quick.Config{Rand: newRand(), MaxCount: quickchecks}); err != nil {
		t.Errorf("Transitivity test for ElNe failed: %v", err)
	}
}
func TestDense_GtScalar(t *testing.T) {
	transFn := func(q *Dense) bool {
		we, _ := willerr(q, ordTypes, nil)
		_, ok := q.Engine().(Gter)
		we = we || !ok

		r := newRand()
		a := q.Clone().(*Dense)
		bv, _ := quick.Value(a.Dtype().Type, r)
		b := bv.Interface()
		c := q.Clone().(*Dense)
		cv, _ := quick.Value(c.Dtype().Type, r)
		c.Memset(cv.Interface())

		axb, err := a.GtScalar(b, true)
		if err, retEarly := qcErrCheck(t, "Gt - a∙b", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		bxc, err := c.GtScalar(b, false)
		if err, retEarly := qcErrCheck(t, "Gt - b∙c", c, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		axc, err := a.Gt(c)
		if err, retEarly := qcErrCheck(t, "Gt - a∙c", a, c, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		ab := axb.Bools()
		bc := bxc.Bools()
		ac := axc.Bools()
		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}

		return true
	}
	if err := quick.Check(transFn, &quick.Config{Rand: newRand(), MaxCount: quickchecks}); err != nil {
		t.Errorf("Transitivity test for Gt failed: %v", err)
	}

}
func TestDense_GteScalar(t *testing.T) {
	transFn := func(q *Dense) bool {
		we, _ := willerr(q, ordTypes, nil)
		_, ok := q.Engine().(Gteer)
		we = we || !ok

		r := newRand()
		a := q.Clone().(*Dense)
		bv, _ := quick.Value(a.Dtype().Type, r)
		b := bv.Interface()
		c := q.Clone().(*Dense)
		cv, _ := quick.Value(c.Dtype().Type, r)
		c.Memset(cv.Interface())

		axb, err := a.GteScalar(b, true)
		if err, retEarly := qcErrCheck(t, "Gte - a∙b", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		bxc, err := c.GteScalar(b, false)
		if err, retEarly := qcErrCheck(t, "Gte - b∙c", c, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		axc, err := a.Gte(c)
		if err, retEarly := qcErrCheck(t, "Gte - a∙c", a, c, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		ab := axb.Bools()
		bc := bxc.Bools()
		ac := axc.Bools()
		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}

		return true
	}
	if err := quick.Check(transFn, &quick.Config{Rand: newRand(), MaxCount: quickchecks}); err != nil {
		t.Errorf("Transitivity test for Gte failed: %v", err)
	}

}
func TestDense_LtScalar(t *testing.T) {
	transFn := func(q *Dense) bool {
		we, _ := willerr(q, ordTypes, nil)
		_, ok := q.Engine().(Lter)
		we = we || !ok

		r := newRand()
		a := q.Clone().(*Dense)
		bv, _ := quick.Value(a.Dtype().Type, r)
		b := bv.Interface()
		c := q.Clone().(*Dense)
		cv, _ := quick.Value(c.Dtype().Type, r)
		c.Memset(cv.Interface())

		axb, err := a.LtScalar(b, true)
		if err, retEarly := qcErrCheck(t, "Lt - a∙b", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		bxc, err := c.LtScalar(b, false)
		if err, retEarly := qcErrCheck(t, "Lt - b∙c", c, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		axc, err := a.Lt(c)
		if err, retEarly := qcErrCheck(t, "Lt - a∙c", a, c, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		ab := axb.Bools()
		bc := bxc.Bools()
		ac := axc.Bools()
		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}

		return true
	}
	if err := quick.Check(transFn, &quick.Config{Rand: newRand(), MaxCount: quickchecks}); err != nil {
		t.Errorf("Transitivity test for Lt failed: %v", err)
	}

}
func TestDense_LteScalar(t *testing.T) {
	transFn := func(q *Dense) bool {
		we, _ := willerr(q, ordTypes, nil)
		_, ok := q.Engine().(Lteer)
		we = we || !ok

		r := newRand()
		a := q.Clone().(*Dense)
		bv, _ := quick.Value(a.Dtype().Type, r)
		b := bv.Interface()
		c := q.Clone().(*Dense)
		cv, _ := quick.Value(c.Dtype().Type, r)
		c.Memset(cv.Interface())

		axb, err := a.LteScalar(b, true)
		if err, retEarly := qcErrCheck(t, "Lte - a∙b", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		bxc, err := c.LteScalar(b, false)
		if err, retEarly := qcErrCheck(t, "Lte - b∙c", c, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		axc, err := a.Lte(c)
		if err, retEarly := qcErrCheck(t, "Lte - a∙c", a, c, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		ab := axb.Bools()
		bc := bxc.Bools()
		ac := axc.Bools()
		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}

		return true
	}
	if err := quick.Check(transFn, &quick.Config{Rand: newRand(), MaxCount: quickchecks}); err != nil {
		t.Errorf("Transitivity test for Lte failed: %v", err)
	}

}
func TestDense_ElEqScalar(t *testing.T) {
	transFn := func(q *Dense) bool {
		we, _ := willerr(q, eqTypes, nil)
		_, ok := q.Engine().(ElEqer)
		we = we || !ok

		r := newRand()
		a := q.Clone().(*Dense)
		bv, _ := quick.Value(a.Dtype().Type, r)
		b := bv.Interface()
		c := q.Clone().(*Dense)
		cv, _ := quick.Value(c.Dtype().Type, r)
		c.Memset(cv.Interface())

		axb, err := a.ElEqScalar(b, true)
		if err, retEarly := qcErrCheck(t, "ElEq - a∙b", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		bxc, err := c.ElEqScalar(b, false)
		if err, retEarly := qcErrCheck(t, "ElEq - b∙c", c, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		axc, err := a.ElEq(c)
		if err, retEarly := qcErrCheck(t, "ElEq - a∙c", a, c, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		ab := axb.Bools()
		bc := bxc.Bools()
		ac := axc.Bools()
		for i, vab := range ab {
			if vab && bc[i] {
				if !ac[i] {
					return false
				}
			}
		}

		return true
	}
	if err := quick.Check(transFn, &quick.Config{Rand: newRand(), MaxCount: quickchecks}); err != nil {
		t.Errorf("Transitivity test for ElEq failed: %v", err)
	}

	symFn := func(q *Dense) bool {
		we, _ := willerr(q, eqTypes, nil)
		_, ok := q.Engine().(ElEqer)
		we = we || !ok

		r := newRand()
		a := q.Clone().(*Dense)
		bv, _ := quick.Value(a.Dtype().Type, r)
		b := bv.Interface()

		axb, err := a.ElEqScalar(b, true)
		if err, retEarly := qcErrCheck(t, "ElEq - a∙b", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		bxa, err := a.ElEqScalar(b, false)
		if err, retEarly := qcErrCheck(t, "ElEq - b∙a", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}
		return reflect.DeepEqual(axb.Data(), bxa.Data())

	}
	if err := quick.Check(symFn, &quick.Config{Rand: newRand(), MaxCount: quickchecks}); err != nil {
		t.Errorf("Symmetry test for ElEq failed: %v", err)
	}
}
func TestDense_ElNeScalar(t *testing.T) {
	symFn := func(q *Dense) bool {
		we, _ := willerr(q, eqTypes, nil)
		_, ok := q.Engine().(ElEqer)
		we = we || !ok

		r := newRand()
		a := q.Clone().(*Dense)
		bv, _ := quick.Value(a.Dtype().Type, r)
		b := bv.Interface()

		axb, err := a.ElNeScalar(b, true)
		if err, retEarly := qcErrCheck(t, "ElNe - a∙b", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		bxa, err := a.ElNeScalar(b, false)
		if err, retEarly := qcErrCheck(t, "ElNe - b∙a", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}
		return reflect.DeepEqual(axb.Data(), bxa.Data())

	}
	if err := quick.Check(symFn, &quick.Config{Rand: newRand(), MaxCount: quickchecks}); err != nil {
		t.Errorf("Symmetry test for ElNe failed: %v", err)
	}
}
func TestDense_GtScalar_assame(t *testing.T) {
	transFn := func(q *Dense) bool {
		we, _ := willerr(q, nonComplexNumberTypes, nil)
		_, ok := q.Engine().(Gter)
		we = we || !ok

		if err := typeclassCheck(q.Dtype(), nonComplexNumberTypes); err != nil {
			return true // we exit early if the generated type is not something we can handle
		}
		r := newRand()
		a := q.Clone().(*Dense)
		bv, _ := quick.Value(a.Dtype().Type, r)
		b := bv.Interface()
		c := q.Clone().(*Dense)
		cv, _ := quick.Value(c.Dtype().Type, r)
		c.Memset(cv.Interface())

		axb, err := a.GtScalar(b, true, AsSameType())
		if err, retEarly := qcErrCheck(t, "Gt - a∙b", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		bxc, err := c.GtScalar(b, false, AsSameType())
		if err, retEarly := qcErrCheck(t, "Gt - b∙c", c, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		axc, err := a.Gt(c, AsSameType())
		if err, retEarly := qcErrCheck(t, "Gt - a∙c", a, c, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		if !threewayEq(axb.Data(), bxc.Data(), axc.Data()) {
			t.Errorf("a: %-v", a)
			t.Errorf("b: %-v", b)
			t.Errorf("c: %-v", c)
			t.Errorf("axb.Data() %v", axb.Data())
			t.Errorf("bxc.Data() %v", bxc.Data())
			t.Errorf("axc.Data() %v", axc.Data())
			return false
		}

		return true
	}
	if err := quick.Check(transFn, &quick.Config{Rand: newRand(), MaxCount: quickchecks}); err != nil {
		t.Errorf("Transitivity test for Gt failed: %v", err)
	}

}
func TestDense_GteScalar_assame(t *testing.T) {
	transFn := func(q *Dense) bool {
		we, _ := willerr(q, nonComplexNumberTypes, nil)
		_, ok := q.Engine().(Gteer)
		we = we || !ok

		if err := typeclassCheck(q.Dtype(), nonComplexNumberTypes); err != nil {
			return true // we exit early if the generated type is not something we can handle
		}
		r := newRand()
		a := q.Clone().(*Dense)
		bv, _ := quick.Value(a.Dtype().Type, r)
		b := bv.Interface()
		c := q.Clone().(*Dense)
		cv, _ := quick.Value(c.Dtype().Type, r)
		c.Memset(cv.Interface())

		axb, err := a.GteScalar(b, true, AsSameType())
		if err, retEarly := qcErrCheck(t, "Gte - a∙b", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		bxc, err := c.GteScalar(b, false, AsSameType())
		if err, retEarly := qcErrCheck(t, "Gte - b∙c", c, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		axc, err := a.Gte(c, AsSameType())
		if err, retEarly := qcErrCheck(t, "Gte - a∙c", a, c, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		if !threewayEq(axb.Data(), bxc.Data(), axc.Data()) {
			t.Errorf("a: %-v", a)
			t.Errorf("b: %-v", b)
			t.Errorf("c: %-v", c)
			t.Errorf("axb.Data() %v", axb.Data())
			t.Errorf("bxc.Data() %v", bxc.Data())
			t.Errorf("axc.Data() %v", axc.Data())
			return false
		}

		return true
	}
	if err := quick.Check(transFn, &quick.Config{Rand: newRand(), MaxCount: quickchecks}); err != nil {
		t.Errorf("Transitivity test for Gte failed: %v", err)
	}

}
func TestDense_LtScalar_assame(t *testing.T) {
	transFn := func(q *Dense) bool {
		we, _ := willerr(q, nonComplexNumberTypes, nil)
		_, ok := q.Engine().(Lter)
		we = we || !ok

		if err := typeclassCheck(q.Dtype(), nonComplexNumberTypes); err != nil {
			return true // we exit early if the generated type is not something we can handle
		}
		r := newRand()
		a := q.Clone().(*Dense)
		bv, _ := quick.Value(a.Dtype().Type, r)
		b := bv.Interface()
		c := q.Clone().(*Dense)
		cv, _ := quick.Value(c.Dtype().Type, r)
		c.Memset(cv.Interface())

		axb, err := a.LtScalar(b, true, AsSameType())
		if err, retEarly := qcErrCheck(t, "Lt - a∙b", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		bxc, err := c.LtScalar(b, false, AsSameType())
		if err, retEarly := qcErrCheck(t, "Lt - b∙c", c, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		axc, err := a.Lt(c, AsSameType())
		if err, retEarly := qcErrCheck(t, "Lt - a∙c", a, c, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		if !threewayEq(axb.Data(), bxc.Data(), axc.Data()) {
			t.Errorf("a: %-v", a)
			t.Errorf("b: %-v", b)
			t.Errorf("c: %-v", c)
			t.Errorf("axb.Data() %v", axb.Data())
			t.Errorf("bxc.Data() %v", bxc.Data())
			t.Errorf("axc.Data() %v", axc.Data())
			return false
		}

		return true
	}
	if err := quick.Check(transFn, &quick.Config{Rand: newRand(), MaxCount: quickchecks}); err != nil {
		t.Errorf("Transitivity test for Lt failed: %v", err)
	}

}
func TestDense_LteScalar_assame(t *testing.T) {
	transFn := func(q *Dense) bool {
		we, _ := willerr(q, nonComplexNumberTypes, nil)
		_, ok := q.Engine().(Lteer)
		we = we || !ok

		if err := typeclassCheck(q.Dtype(), nonComplexNumberTypes); err != nil {
			return true // we exit early if the generated type is not something we can handle
		}
		r := newRand()
		a := q.Clone().(*Dense)
		bv, _ := quick.Value(a.Dtype().Type, r)
		b := bv.Interface()
		c := q.Clone().(*Dense)
		cv, _ := quick.Value(c.Dtype().Type, r)
		c.Memset(cv.Interface())

		axb, err := a.LteScalar(b, true, AsSameType())
		if err, retEarly := qcErrCheck(t, "Lte - a∙b", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		bxc, err := c.LteScalar(b, false, AsSameType())
		if err, retEarly := qcErrCheck(t, "Lte - b∙c", c, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		axc, err := a.Lte(c, AsSameType())
		if err, retEarly := qcErrCheck(t, "Lte - a∙c", a, c, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		if !threewayEq(axb.Data(), bxc.Data(), axc.Data()) {
			t.Errorf("a: %-v", a)
			t.Errorf("b: %-v", b)
			t.Errorf("c: %-v", c)
			t.Errorf("axb.Data() %v", axb.Data())
			t.Errorf("bxc.Data() %v", bxc.Data())
			t.Errorf("axc.Data() %v", axc.Data())
			return false
		}

		return true
	}
	if err := quick.Check(transFn, &quick.Config{Rand: newRand(), MaxCount: quickchecks}); err != nil {
		t.Errorf("Transitivity test for Lte failed: %v", err)
	}

}
func TestDense_ElEqScalar_assame(t *testing.T) {
	transFn := func(q *Dense) bool {
		we, _ := willerr(q, nonComplexNumberTypes, nil)
		_, ok := q.Engine().(ElEqer)
		we = we || !ok

		if err := typeclassCheck(q.Dtype(), nonComplexNumberTypes); err != nil {
			return true // we exit early if the generated type is not something we can handle
		}
		r := newRand()
		a := q.Clone().(*Dense)
		bv, _ := quick.Value(a.Dtype().Type, r)
		b := bv.Interface()
		c := q.Clone().(*Dense)
		cv, _ := quick.Value(c.Dtype().Type, r)
		c.Memset(cv.Interface())

		axb, err := a.ElEqScalar(b, true, AsSameType())
		if err, retEarly := qcErrCheck(t, "ElEq - a∙b", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		bxc, err := c.ElEqScalar(b, false, AsSameType())
		if err, retEarly := qcErrCheck(t, "ElEq - b∙c", c, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		axc, err := a.ElEq(c, AsSameType())
		if err, retEarly := qcErrCheck(t, "ElEq - a∙c", a, c, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		if !threewayEq(axb.Data(), bxc.Data(), axc.Data()) {
			t.Errorf("a: %-v", a)
			t.Errorf("b: %-v", b)
			t.Errorf("c: %-v", c)
			t.Errorf("axb.Data() %v", axb.Data())
			t.Errorf("bxc.Data() %v", bxc.Data())
			t.Errorf("axc.Data() %v", axc.Data())
			return false
		}

		return true
	}
	if err := quick.Check(transFn, &quick.Config{Rand: newRand(), MaxCount: quickchecks}); err != nil {
		t.Errorf("Transitivity test for ElEq failed: %v", err)
	}

	symFn := func(q *Dense) bool {
		we, _ := willerr(q, nonComplexNumberTypes, nil)
		_, ok := q.Engine().(ElEqer)
		we = we || !ok

		if err := typeclassCheck(q.Dtype(), nonComplexNumberTypes); err != nil {
			return true // we exit early if the generated type is not something we can handle
		}
		r := newRand()
		a := q.Clone().(*Dense)
		bv, _ := quick.Value(a.Dtype().Type, r)
		b := bv.Interface()

		axb, err := a.ElEqScalar(b, true, AsSameType())
		if err, retEarly := qcErrCheck(t, "ElEq - a∙b", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		bxa, err := a.ElEqScalar(b, false, AsSameType())
		if err, retEarly := qcErrCheck(t, "ElEq - b∙a", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}
		return reflect.DeepEqual(axb.Data(), bxa.Data())

	}
	if err := quick.Check(symFn, &quick.Config{Rand: newRand(), MaxCount: quickchecks}); err != nil {
		t.Errorf("Symmetry test for ElEq failed: %v", err)
	}
}
func TestDense_ElNeScalar_assame(t *testing.T) {
	symFn := func(q *Dense) bool {
		we, _ := willerr(q, nonComplexNumberTypes, nil)
		_, ok := q.Engine().(ElEqer)
		we = we || !ok

		if err := typeclassCheck(q.Dtype(), nonComplexNumberTypes); err != nil {
			return true // we exit early if the generated type is not something we can handle
		}
		r := newRand()
		a := q.Clone().(*Dense)
		bv, _ := quick.Value(a.Dtype().Type, r)
		b := bv.Interface()

		axb, err := a.ElNeScalar(b, true, AsSameType())
		if err, retEarly := qcErrCheck(t, "ElNe - a∙b", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}

		bxa, err := a.ElNeScalar(b, false, AsSameType())
		if err, retEarly := qcErrCheck(t, "ElNe - b∙a", a, b, we, err); retEarly {
			if err != nil {
				return false
			}
			return true
		}
		return reflect.DeepEqual(axb.Data(), bxa.Data())

	}
	if err := quick.Check(symFn, &quick.Config{Rand: newRand(), MaxCount: quickchecks}); err != nil {
		t.Errorf("Symmetry test for ElNe failed: %v", err)
	}
}