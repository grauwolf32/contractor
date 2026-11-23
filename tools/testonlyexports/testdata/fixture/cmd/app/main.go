// Command app is the production code of the fixture.
package main

import (
	"fmt"

	"example.com/fixture/core"
)

// Helper is called only by this command's test.
func Helper() {}

func main() {
	core.Used()
	fmt.Println(core.NewStore().Get(), core.NewReader().Read(), core.Named{}, core.Impl{}, core.NewBox(1).Value())
	var pinger core.Pinger = core.Whole{}
	pinger.Ping()
	fmt.Println(pinger.Name(), core.CodeA)
}
