import std;

import nanogui.sdlbackend : SdlBackend;
import nanogui.widget : Widget;
import nanogui.glcanvas : GLCanvas;

struct Vertex
{
	import nanogui.common;
	Vector3f position;
	Vector3f color;
}

class MyGlCanvas : GLCanvas
{
	this(Widget parent, int w, int h)
	{
		super(parent, w, h);

		const program_source = 
			"#version 130

			#if VERTEX_SHADER
			uniform mat4 modelViewProj;
			in vec3 position;
			in vec3 color;
			out vec4 frag_color;
			void main() {
				frag_color  = vec4(color, 1.0);
				gl_Position = modelViewProj * vec4(position, 1.0);
			}
			#endif

			#if FRAGMENT_SHADER
			out vec4 color;
			in vec4 frag_color;
			void main() {
				color = frag_color;
			}
			#endif";

		_program = new GLProgram(program_source);
		assert(_program);
		auto _vert_spec = scoped!(VertexSpecification!Vertex)(_program);

		_vao_axis = scoped!GLVAO();
		{
			_idx_axis = [ 1, 0, 2, ];

			auto vertices = 
			[
				Vertex(Vector3f(0,  0, 0), Vector3f(1, 0, 0)),
				Vertex(Vector3f(1,  0, 0), Vector3f(0, 1, 0)),
				Vertex(Vector3f(0,  1, 0), Vector3f(0, 0, 1)),
			];

			_buf_axis = scoped!GLBuffer(GL_ARRAY_BUFFER, GL_STATIC_DRAW, vertices);
			auto ibo = scoped!GLBuffer(GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW, _idx_axis);

			_vao_axis.bind();
			_buf_axis.bind();
			ibo.bind();
			_vert_spec.use();
			_vao_axis.unbind();
		}

		_vao_data = scoped!GLVAO();
		{
			auto vertices = Vertex(Vector3f(.5,  .5, 0), Vector3f(1, 1, 1)).repeat(DataSize).array;

			_buf_data = scoped!GLBuffer(GL_ARRAY_BUFFER, GL_STATIC_DRAW, vertices);
			_vao_data.bind();
			_buf_data.bind();
			_vert_spec.use();
			_vao_data.unbind();
		}
	}

	override void drawGL()
	{
		auto mvp = mat4f.identity * scale;
		mvp.ptr[3*4+3] = 1;

		GLboolean depth_test_enabled;
		glGetBooleanv(GL_DEPTH_TEST, &depth_test_enabled);
		if (!depth_test_enabled)
			glEnable(GL_DEPTH_TEST);
		scope(exit)
		{
			if (!depth_test_enabled)
				glDisable(GL_DEPTH_TEST);
		}

		_program.uniform("modelViewProj").set(mvp);
		_program.use();
		scope(exit) _program.unuse();

		_vao_axis.bind();
		glDrawElements(GL_LINE_STRIP, castFrom!ulong.to!int(_idx_axis.length), GL_UNSIGNED_INT, cast(void *) 0);
		_vao_axis.unbind();

		_vao_data.bind();
		glDrawArrays(GL_POINTS, 0, DataSize);
		_vao_data.unbind();
	}
	float scale = 1;

private:
	GLProgram _program;

	import std.typecons : scoped;
	import gfm.opengl;
	import gfm.math;
	import nanogui.common;
	import nanogui.widget : Widget;

	alias ScopedGLBuffer = typeof(scoped!GLBuffer(GL_ARRAY_BUFFER, GL_STATIC_DRAW, Vertex[].init));
	alias ScopedGLVAO = typeof(scoped!GLVAO());
	ScopedGLVAO _vao_axis, _vao_data;
	int[] _idx_axis;
	ScopedGLBuffer _buf_axis, _buf_data;

	enum DataSize = 30_000;
}

class MyGui : SdlBackend
{
	import nanogui.screen : Screen;
	import nanogui.textbox : FloatBox;
	import nanogui.window : Window;
	import nanogui.common : Vector2i, Vector2f, Color;
	import nanogui.layout : BoxLayout, Orientation, GroupLayout;
	import nanogui.button : Button;

	FloatBox!float[2][2] covmat, eig_vect;
	FloatBox!float[2] eig_val;
	FloatBox!float angle;
	MyGlCanvas glcanvas;
	Vector2f[] samples;
	union
	{
		float[4] sbase;
		struct
		{
			float sx, sxy, sxy2, sy;
		}
	}

	this(int w, int h, string title)
	{
		super(w, h, title);

		sx = 4;
		sxy = sxy2 = 1;
		sy = 4;
		samples.length = glcanvas.DataSize;
	}

	auto generateData() @nogc
	{
		import mir.ndslice.slice: sliced;
		import mir.random;
		import mir.random.ndvariable : multivariateNormalVar, MultivariateNormalVariable;
		import mir.random.variable : normalVar;

		scope Random* gen = threadLocalPtr!Random;
		gen.__ctor(0);
		// float[4] dummy = sbase[];
		// auto sigma = dummy[].sliced(2,2);
		// if (!MultivariateNormalVariable!float.cholesky(sigma))
		// 	return false;
		// auto rv = multivariateNormalVar!float(sigma, true);
		// foreach(i; 0..glcanvas.DataSize)
		// {
		// 	rv(gen, samples[i].v[]);
		// }

		auto sigma_angle = 10.0*PI/180.0;
		auto sigma_range = 0.100;
		auto ang = normalVar!float(sigma_angle);
		auto rng = normalVar!float(sigma_range);
		foreach(i; 0..glcanvas.DataSize)
		{
			auto angle = ang(gen);
			auto range = rng(gen);
			// angle += PI/4;
			// range += 150000;
			samples[i].x = angle;//range * sin(angle);
			samples[i].y = range;// * cos(angle);
// debug
// {
// 	import std;
// 	writeln(samples[i]);
// 	// break;
// }
		}

		debug
		{
			import dstats.summary : MeanSD;
			// MeanSD summ_x, summ_y;
			import std;

			import dstats.cor;
			// samples.map!"a.x".each!(v=>summ_x.put(v));
			// samples.map!"a.y".each!(v=>summ_y.put(v));
			// writeln(summ_x);
			// writeln(summ_y);
			writeln(mean(samples.map!(a=>a.x))*180/PI);
			writeln(covariance(samples.map!(a=>a.x), samples.map!(a=>a.y)));
			writeln(mean(samples.map!(a=>a.y)));
			// auto input = [[1, 0], [2, 1], [1, 3], [1, 4]];
			// auto X = input.map!"a[0]";
			// auto Y = input.map!"a[1]";
			// const xe = X.sum/cast(double) input.length;
			// const ye = Y.sum/cast(double) input.length;
			// writeln("xe: ", xe);
			// writeln("ye: ", ye);
			// // E(x - E(x)*(x - E(x)))
			// auto xx = 0.0;
			// foreach(x; input.map!"a[0]")
			// {
			// 	xx += (x - xe)^^2;
			// 	writeln(xx, ": x:", x, " (x-xe):", x-xe);
			// }
			// xx /= (input.length-1);
			
			// // E(x - E(x)*(y - E(y)))
			// auto xy = 0.0;
			// foreach(a; input)
			// {
			// 	const x = a[0];
			// 	const y = a[1];
			// 	xy += (x - xe)*(y - ye);
			// }
			// xy /= (input.length-1);
			// // E(y - E(y)*(y - E(y)))
			// auto yy = 0.0;
			// foreach(y; input.map!"a[1]")
			// {
			// 	yy += (y - ye)^^2;
			// }
			// yy /= (input.length-1);
			// // writeln("xx: ", xx);
			// // writeln("xy: ", xy);
			// // writeln("yy: ", yy);
			// writefln("( %s, %s )", xx, xy);
			// writefln("( %s, %s )", xy, yy);
			auto covMatrix = [[0.0], [0.0, 0.0]];
			auto input = [[1, 2, 1, 1], [0, 1, 3, 4]];
			covarianceMatrix(input, covMatrix);
			writeln(covMatrix);
		}

		return true;
	}

	auto updateCovarianceMatrix()
	{
		sx   = covmat[0][0].value;
		sxy  = covmat[0][1].value;
		sxy2 = covmat[1][0].value;
		sy   = covmat[1][1].value;
	}

	auto updateData()
	{
		import std;
		import mir.ndslice : sliced;
		import lubeck;

		// enum theta = 0 * PI / 180.0;
		// auto a = [
		// cos(theta), -sin(theta),
		// sin(theta),  cos(theta)].sliced(2, 2);
		// auto a = [
		// 0.707107, 0.707107,
		// 0.707107, 0.707107].sliced(2, 2);

		updateCovarianceMatrix;
		if (!generateData)
			return false;

		auto a = [
			sx,  sxy,
			sxy, sy
		].dup.sliced(2, 2);

		auto eigr = eigSymmetric('L', a);

		// writeln(eigr.values);
		// writeln(eigr.vectors);

		eig_vect[0][0].value = eigr.vectors[0][0];
		eig_vect[0][1].value = eigr.vectors[0][1];
		eig_vect[1][0].value = eigr.vectors[1][0];
		eig_vect[1][1].value = eigr.vectors[1][1];

		eig_val[0].value = eigr.values[0];
		eig_val[1].value = eigr.values[1];

		import std.math : PI;
		import nanogui.common : Vector3f;
		import gfm.math : angleBetween;

		enum north = Vector3f(0, 1, 0);
		auto v3 = Vector3f(0, 0, 0);
		v3.ptr[0..2] = eigr.vectors[0].field;
		v3.ptr[3] = 0;
		angle.value = 180 * angleBetween(north, v3) / PI;

		const O = Vector3f(0, 0, 0);
		const X = Vector3f(eigr.vectors[0][0]*eigr.values[0], eigr.vectors[0][1]*eigr.values[0], 0);
		const Y = Vector3f(eigr.vectors[1][0]*eigr.values[1], eigr.vectors[1][1]*eigr.values[1], 0);
		glcanvas._buf_axis.setData([
			Vertex(O, Vector3f(1, 0, 0)),
			Vertex(X, Vector3f(0, 1, 0)),
			Vertex(Y, Vector3f(0, 0, 1)),
		]);

		glcanvas._buf_data.setData(
			samples.map!((ref v) {
				return Vertex(Vector3f(v, 0), Vector3f(0.35, 0.35, 0.15));
			}).array
		);

		glcanvas.scale = 1.0/10;

		return true;
	}

	override void onVisibleForTheFirstTime()
	{
		{
			auto window = new Window(screen, "2D covariance matrix visualisation");
			window.position = Vector2i(360, 20);
			window.layout = new GroupLayout();
			glcanvas = new MyGlCanvas(window, 250, 250);
			glcanvas.size = Vector2i(160, 160);
			glcanvas.backgroundColor = Color(0.1f, 0.1f, 0.1f, 1.0f);
			glcanvas = new MyGlCanvas(window, 250, 250);
			glcanvas.size = Vector2i(160, 160);
			glcanvas.backgroundColor = Color(0.1f, 0.1f, 0.1f, 1.0f);
			new FloatBox!float(window);
		}

		{
			auto window = new Window(screen, "Input data");
			window.position(Vector2i(20, 20));
			window.layout = new GroupLayout();

			import nanogui.label : Label;
			new Label(window, "2D covariance matrix");
			{
				foreach(x; 0..2)
				{
					auto cov = new Widget(window);
					cov.layout = new BoxLayout(Orientation.Horizontal);

					foreach(y; 0..2)
					{
						auto fb = new FloatBox!float(cov);
						fb.fixedWidth = 90;
						fb.editable = true;
						fb.spinnable = true;
						fb.value = (x == y) ? 1 : 0;
						covmat[x][y] = fb;
					}
				}
				with(covmat[1][0])
				{
					editable = false;
					spinnable = false;
				}
				covmat[0][1].callback = (float value) {
					auto old = covmat[1][0].value;
					covmat[1][0].value = value;
					if (!updateData)
					{
						covmat[0][1].value = old;
						covmat[1][0].value = old;
					}
				};

				covmat[0][0].callback = (float value) {
					// TODO this code doesn't work because value has been updated
					// and old value is lost actually
					auto old = covmat[0][0].value;
					if (!updateData)
					{
						covmat[0][0].value = old;
					}
				};

				covmat[1][1].callback =  (float value) {
					// TODO this code doesn't work because value has been updated
					// and old value is lost actually
					auto old = covmat[1][1].value;
					if (!updateData)
					{
						covmat[1][1].value = old;
					}
				};

				covmat[0][0].value = sx;
				covmat[0][1].value = sxy;
				covmat[1][0].value = sxy;
				covmat[1][1].value = sy;
			}

			new Label(window, "Eigen vectors");
			{
				foreach(x; 0..2)
				{
					auto vectors = new Widget(window);
					vectors.layout = new BoxLayout(Orientation.Horizontal);

					foreach(y; 0..2)
					{
						auto fb = new FloatBox!float(vectors);
						fb.fixedWidth = 90;
						fb.value = float.nan;
						eig_vect[x][y] = fb;
					}
				}
			}

			new Label(window, "Eigen values");
			{
				auto values = new Widget(window);
				values.layout = new BoxLayout(Orientation.Horizontal);
				eig_val[0] = new FloatBox!float(values);
				eig_val[0].fixedWidth = 90;
				eig_val[1] = new FloatBox!float(values);
				eig_val[1].fixedWidth = 90;
			}

			new Label(window, "Angle (Degree)");
			{
				auto w = new Widget(window);
				w.layout = new BoxLayout(Orientation.Horizontal);
				angle = new FloatBox!float(w);
				angle.tooltip = "The angle between Y (green) axis and north";
				angle.fixedWidth = 90;
			}
		}

		updateData;

		// now we should do layout manually yet
		screen.performLayout(ctx);
	}
}

void main()
{
	// import mir.ndslice;
	// import lubeck;
	// import std.stdio : writefln;

	// enum fmt = "%(%(%.2f %)\n%)\n";

	// auto matrix = slice!double(4, 5);
	// //writefln(fmt, matrix);

	// matrix[] = 0;
	// matrix.diagonal[] = 1;
	// matrix.transposed[3][] = 1;
	// matrix.transposed[3].writeln;

	// matrix[0][] = [1, 0, 0, 0, 2];
	// matrix[1][] = [0, 0, 3, 0, 0];
	// matrix[2][] = [0, 0, 0, 0, 0];
	// matrix[3][] = [0, 4, 0, 0, 0];

	// writeln("matrix:");
	// writefln(fmt, matrix);

	// auto s = svd(matrix);
	// writeln("s.u: ");
	// writefln(fmt, s.u);
	// writeln("s.sigma: ");
	// writeln(s.sigma);
	// writeln("s.vt: ");
	// writefln(fmt, s.vt);

	// {
	// 	auto t = s.sigma.shape[0];
	// 	auto sigma = slice!double(matrix.shape);
	// 	sigma[] = 0;
	// 	sigma.diagonal[] = s.sigma;
	// 	/*writeln("sigma:");
	// 	writefln(fmt, sigma);
	// 	auto a = s.u.mtimes(sigma);
	// 	writeln("a: ");
	// 	writefln(fmt, a);
	// 	auto b = a.mtimes(s.vt);
	// 	writefln(fmt, b);*/
	// 	writefln(fmt, s.u.mtimes(sigma).mtimes(s.vt));
	// }

	{
		auto gui = new MyGui(1000, 800, "Eigenvalues & eigenvectors");
		gui.run();
	}

}