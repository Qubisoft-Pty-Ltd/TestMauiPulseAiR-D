using Microsoft.Maui.Devices.Sensors;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using Tensorflow;
using Tensorflow.Util;
using Tensorflow.NumPy;
using static Tensorflow.Binding;
using Tensorflow.Keras.Layers;

namespace TestMauiApp.Source;

public class SciSharpExample
{
    public ExampleConfig Config { get; set; }
/*    protected LayersApi layers = new LayersApi();*/

    public virtual void BuildModel()
    {

    }

    public virtual Graph BuildGraph()
    {
        throw new NotImplementedException();
    }

    public virtual Graph ImportGraph()
    {
        throw new NotImplementedException();
    }

    public virtual void PrepareData()
    {
        throw new NotImplementedException();
    }

    public virtual void Train()
    {

    }

    public virtual void Test()
    {

    }

    public virtual void Predict()
    {
    }

    public virtual string FreezeModel()
    {
        throw new NotImplementedException();
    }
}