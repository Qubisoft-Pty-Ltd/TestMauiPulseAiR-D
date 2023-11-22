using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TestMauiApp.Source
{
    public class ExampleConfig
    {
        /// <summary>
        /// Example name
        /// </summary>
        public string Name { get; set; }

        /// <summary>
        /// True to run example
        /// </summary>
        public bool Enabled { get; set; } = true;

        /// <summary>
        /// Set true to import the computation graph instead of building it.
        /// </summary>
        public bool IsImportingGraph { get; set; } = false;
    }
}
