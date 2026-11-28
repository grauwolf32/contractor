package runtimewire

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/grauwolf32/contractor/internal/contracts/contractstest"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protoreflect"
	"google.golang.org/protobuf/reflect/protoregistry"
)

func TestSharedAllocationProtobufShapes(t *testing.T) {
	var cases struct {
		Valid, Invalid []struct {
			Type  string
			Value json.RawMessage
		}
	}
	if err := json.Unmarshal(contractstest.ReadFile(t, "api", "testdata", "v1alpha1", "grpc-allocation-cases.json"), &cases); err != nil {
		t.Fatal(err)
	}
	if len(cases.Valid) < 18 || len(cases.Invalid) < 6 {
		t.Fatal("incomplete allocation fixtures")
	}
	for _, group := range []struct {
		valid bool
		cases []struct {
			Type  string
			Value json.RawMessage
		}
	}{{true, cases.Valid}, {false, cases.Invalid}} {
		for _, test := range group.cases {
			t.Run(test.Type, func(t *testing.T) {
				typ, err := protoregistry.GlobalTypes.FindMessageByName(protoreflect.FullName("contractor.runtime.v1alpha1." + test.Type))
				if err != nil {
					t.Fatal(err)
				}
				request := typ.New().Interface()
				err = EncodeDomain(test.Value, request, 8<<20, (8<<20)+1024)
				if !group.valid {
					if err == nil {
						t.Fatal("invalid value encoded")
					}
					return
				}
				if err != nil {
					t.Fatal(err)
				}
				binary, err := proto.Marshal(request)
				if err != nil {
					t.Fatal(err)
				}
				received := typ.New().Interface()
				if err = proto.Unmarshal(binary, received); err != nil {
					t.Fatal(err)
				}
				var actual, expected any
				if err = DecodeDomain(received, &actual, 8<<20, (8<<20)+1024); err != nil {
					t.Fatal(err)
				}
				if err = json.Unmarshal(test.Value, &expected); err != nil {
					t.Fatal(err)
				}
				if !reflect.DeepEqual(actual, expected) {
					t.Fatal("wire changed domain shape or presence")
				}
				received.ProtoReflect().SetUnknown([]byte{0xf8, 0x07, 0x01})
				if DecodeDomain(received, &actual, 8<<20, (8<<20)+1024) == nil {
					t.Fatal("unknown wire field accepted")
				}
			})
		}
	}
}
