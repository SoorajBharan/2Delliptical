#ifndef ELLIPTICAL
#define ELLIPTICAL

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_dgp.h>
#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/fe/fe.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <iostream>
#include <fstream>
#include <cmath>


namespace ELLIPTICAL
{
using namespace dealii;
/////////////////////////////////////class for calculating Dirichlet Boundary Conditions/////////////////////////////////
template <int dim>
class DCBoundaryValueQ : public Function<dim>
{
	public:
		virtual double value(const Point<2> &p)const override
		{
			double x = p[0];
			double y = p[1];
			double c = 1;
			double value;
			value=sin(c*M_PI*x)*sin(c*M_PI*y);
			return value;
		}
};
///////////////////////////////////////class for calculation Neumann Boundary conditions/////////////////////////////
template <int dim>
class NUBoundaryValueQ : public Function<dim>
{
	public:
		virtual double value(const Point<2> &p)const override
		{
			double x = p[0],y = p[1],c = 1;
			double value;
			value = c*M_PI*sin(c*M_PI*(x+y));
			return value;

		}
};
/////////////////////////////////////class for calculating the value for f(x)///////////////////////////////////////
template <int dim>
class RHSValue : public Function<dim>
{
	public:
		virtual double value(const Point<dim> &p)const override
		{
			double x = p[0];
			double y = p[1];
			double value,c;
			c=1;
			value = -2*pow(c*M_PI,2)*sin(c*M_PI*x)*sin(c*M_PI*y);
			return value;
		}

};

template <int dim>
class elliptical
{
public:
	using cell_iterator = typename DoFHandler<dim>::active_cell_iterator;
	elliptical();
	void run();
private:
	void make_grid();
	void setup_system();
	void assemble_system();
//	void solve();
	Vector<double> volume_integral(Vector<double> q,const uint,FEValues<dim>);
	Vector<double> flux_integral(Vector<double>,const uint,FEValues<dim>);
	Vector<double> gradient_calc(const cell_iterator &cell,FEValues<dim>,Vector<double> q,uint,uint,uint);
	Vector<double> value_calc(const cell_iterator &cell,FEValues<dim>,Vector<double> q,uint,uint);

	FullMatrix<double> vector_to_matrix(const uint, FEValues<dim>,int);

	void output_data();

	Triangulation<dim>	tria;
	FE_DGQ<dim>		fe;
	DoFHandler<dim>		dof_handler;

	SparsityPattern 	sparsity_pattern;
	SparseMatrix<double>	system_mass_matrix;
//	SparseMatrix<double>	system_diff_matrix;
//	SparseMatrix<double>	system_flux_matrix;
		
	FullMatrix<double>	system_vol_matrix;
	FullMatrix<double>	system_flux_matrix;

	Vector<double>		q_sol;

	int 	no_dofs;
};//end of elliptical class template declaration

///////////////CONSTRUCTOR OF THE CLASS/////////////////////////////////////////

template <int dim>
elliptical<dim>::elliptical()
	:fe(1),dof_handler(tria)
{}

/////////////////////////FUNCTION FOR CREATING THE GRID/////////////////////////

template <int dim>
void elliptical<dim>::make_grid()
{
	GridGenerator::hyper_cube(tria,-1,1);
	tria.refine_global(4);
}

////////////////////////FUNCTION OF INITIALISING ALL THE MATRICES AND VECTORS///////////////

template <int dim>
void elliptical<dim>::setup_system()
{
	dof_handler.distribute_dofs(fe);
	no_dofs=dof_handler.n_dofs();

	DynamicSparsityPattern dsp(dof_handler.n_dofs(),
			    	   dof_handler.n_dofs());
	DoFTools::make_sparsity_pattern(dof_handler,dsp);
	sparsity_pattern.copy_from(dsp);

	system_mass_matrix.reinit(sparsity_pattern);
//	system_diff_matrix.reinit(sparsity_pattern);
	
	q_sol.reinit(no_dofs);

	system_vol_matrix.reinit(no_dofs,no_dofs);
	system_flux_matrix.reinit(no_dofs,no_dofs);
}

////////////////////ASSEMBLE SYSTEM FUNCTION/////////////////////

template <int dim>
void elliptical<dim>::assemble_system()
{
	QGaussLobatto<dim>	quadrature_formula(fe.get_degree()+2);
	QGaussLobatto<dim-1>	face_quadrature_formula(fe.get_degree()+2);

	FEValues<dim>		fe_values(fe,
			  	quadrature_formula,
			  	update_values | 
			  	update_gradients |
			  	update_quadrature_points | update_JxW_values);

	FEFaceValues<dim>		fe_face_values(fe,
			  	face_quadrature_formula,
			  	update_values | 
			  	update_gradients |
			  	update_quadrature_points | update_JxW_values);

	const uint dofs_per_cell = fe.n_dofs_per_cell();

	std::vector<types::global_dof_index>	local_dof_indices(dofs_per_cell);

	const uint n_q_points = quadrature_formula.size();

	FullMatrix<double>	elem_mass_mat(dofs_per_cell,dofs_per_cell);

	//Calculation for elemental mass matrix
	for (const auto &cell : dof_handler.active_cell_iterators())
	{
		elem_mass_mat = 0;
		fe_values.reinit(cell);

		for(uint q_points=0; q_points<n_q_points; ++q_points)
		{
			for(uint i = 0; i<dofs_per_cell; ++i)
			{
				for(uint j=0;j<dofs_per_cell;j++)
				{
					elem_mass_mat(i,j)+= fe_values.shape_value(i,q_points)*//phi(i,k)
      								fe_values.shape_value(j,q_points)*//phi(j,k)
      								fe_values.JxW(q_points);
				}
			}
		}
		
		//populate local_dof_index with global dofs for the current cell
		cell->get_dof_indices(local_dof_indices);

		//Constructing global mass matrix from DSS operation
		for(uint i=0;i<dofs_per_cell;++i)
		{
			for(uint j=0;j<dofs_per_cell;++j)
			{
				system_mass_matrix.add(local_dof_indices[i],local_dof_indices[j],elem_mass_mat(i,j));
			}
		}
	}

	system_vol_matrix=vector_to_matrix(n_q_points,fe_values,0);
	system_flux_matrix=vector_to_matrix(n_q_points,fe_values,1);

}

///////////////FUNCTION FOR FINDING VOLUME INTERGRAL IN SIPG /////////////////////

template <int dim>
Vector<double> elliptical<dim>::volume_integral(Vector<double> q, const uint n_q_points, FEValues<dim> fe_V)
{
	Vector<double>		R;
	R.reinit(no_dofs);

	const uint 		dofs_per_cell = fe.n_dofs_per_cell();

	Tensor<1,dim>	grad;

	std::vector<types::global_dof_index> local_dof_index(dof_handler.n_dofs());


	for (const auto &cell : dof_handler.active_cell_iterators())
	{
		//populate the local_dof_index with global dof index for the current cell.
		cell->get_dof_indices(local_dof_index);
		for(uint q_points = 0; q_points < n_q_points; ++q_points)
		{
			grad[0]=gradient_calc(cell,fe_V,q,q_points,dofs_per_cell,0);
			grad[1]=gradient_calc(cell,fe_V,q,q_points,dofs_per_cell,1);

			for (uint i = 0; i < dofs_per_cell; ++i)
			{
				R[local_dof_index[i]] += (fe_V.shape_grad_component(i,q_points,0)*
      								grad[0]*
								fe_V.JxW(q_points))+
								(fe_V.shape_grad_component(i,q_points,1)*
								grad[1]*
								fe_V.JxW(q_points));
			}
		}
	}
	return R;
}

////////////////FUNCTION FOR FINDING SURFACE INTEGRAL IN SIPG/////////////////////////////

template <int dim>
Vector<double> elliptical<dim>::flux_integral(Vector<double> q, const uint n_q_points, FEValues<dim> fe_V)
{
	
	Vector<double>		R;
	R.reinit(no_dofs);

//	const uint 		dofs_per_cell = fe.n_dofs_per_cell();
	Tensor<1,dim> 		grad_qL,grad_qR,grad_flux;

	Vector<double>		qL,qR;
	qL.reinit(n_q_points);
	qR.reinit(n_q_points);

	Vector<double>		q_flux;
	q_flux.reinit(n_q_points);
	
	const uint dofs_per_face = fe.n_dofs_per_face();

	std::vector<types::global_dof_index> local_dof_index(dof_handler.n_dofs());

	for (const auto &face : tria.active_face_iterators())
	{
		if(face->at_boundary())
		{
			std::cout<<"Boundary"<<std::endl;
			
		}
      		else
		{
			//Identify left and right faces
			auto const &left_cell = face->neighbor(0);
			auto const &right_cell = face->neighbour(0);
			//Find the face normal from left cell where the face islocated
			const Point<2> face_normal = left_cell->face(face->face_index())->normal_vector();
			//Loop over integration points	
			for (uint q_points = 0; q_points<n_q_points;++q_points)
			{
				grad_qL[0]=gradient_calc(left_cell,fe_V,q,q_points,dofs_per_face,0);
				grad_qL[1]=gradient_calc(left_cell,fe_V,q,q_points,dofs_per_face,1);
				grad_qR[0]=gradient_calc(right_cell,fe_V,q,q_points,dofs_per_face,0);
				grad_qR[1]=gradient_calc(left_cell,fe_V,q,q_points,dofs_per_face,1);

				qL[q_points]=value_calc(left_cell,fe_V,q,q_points,dofs_per_face);
				qR[q_points]=value_calc(right_cell,fe_V,q,q_points,dofs_per_face);

				grad_flux[0]=0.5*(grad_qR[0]+grad_qL[0]-face_normal[0]*(qR[q_points]-qL[q_points]));
				grad_flux[1]=0.5*(grad_qR[1]+grad_qL[1]-face_normal[1]*(qR[q_points]-qL[q_points]));
					
				q_flux[q_points]=0.5*(qL[q_points]+qR[q_points]);

				//Contribution of face on left element
				for (uint i = 0; i < dofs_per_face; i++)
				{
					left_cell->get_dof_indices(local_dof_index);
					R[local_dof_index[i]] += fe_V.JxW*		
      									fe_V.shape_value(i,q_points)*
      									(face_normal[0]*grad_flux[0]+face_normal[1]*grad_flux[1]);
					R[local_dof_index[i]]+=fe_V.JxW*
      									(qL[q_points]-q_flux[q_points])*
      									(face_normal[0]*fe_V.shape_grad_component(i,q_points,0)+
										face_normal[1]*fe_V.shape_grad_component(i,q_points,1));

				}
				//Contribution of face on right element
				for (uint i = 0; i<dofs_per_face; ++i)
				{
					right_cell->get_dof_indices(local_dof_index);
					R[local_dof_index[i]] -= fe_V.JxW*
      									fe_V.shape_value(i,q_points)*
      									(face_normal[0]*grad_flux[0]+face_normal[1]*grad_flux[1]);
					R[local_dof_index[i]] -= fe_V.JxW*
      									(qR[q_points]-q_flux[q_points])*
      									(face_normal[0]*fe_V.shape_grad_component(i,q_points,0)+
									face_normal[1]*fe_V.shape_grad_component(i,q_points,1));
      									
				}
			}
		}

	}
	return R;

}
/////////////////////////////////////////IMPLEMENTATION OF VECTOR TO MATRIX FUNCTION////////////////////////////////////////////////////
template <int dim>
FullMatrix<double> elliptical<dim>::vector_to_matrix(const uint n_q_points, FEValues<dim> fe_V,int flag)
{
	FullMatrix<double>	M;
	M.reinit(no_dofs,no_dofs);

	Vector<double>		temp_q,R;
	temp_q.reinit(no_dofs);
	R.reinit(no_dofs);

	for(uint i = 0; i < no_dofs; ++i)
	{
		temp_q=0.0;
		temp_q[i]=1;
		if(flag==0)
		{R=volume_integral(temp_q,n_q_points,fe_V);}
		else{R=flux_integral(temp_q,n_q_points,fe_V);}

		for(uint m=0;m<no_dofs;++m)
		{
			M(m,i)=R[m];
		}
	}
	return M;
}

//////////////////////////////////////////FUNCTION FOR CALCULATION GRADIENT OVER THE CELL//////////////////////
template <int dim>
Vector<double> elliptical<dim>::gradient_calc(const cell_iterator &cell,FEValues<dim> fe_V, Vector<double>q,uint q_point, uint dof,uint component)
{
	Vector<double>		grad;
	grad.reinit(dof);
	
	std::vector<types::global_dof_index> local_dof_index(dof_handler.n_dofs());
	cell->get_dof_indices(local_dof_index);
	grad=0;
	for(uint i = 0; i<dof;++i)
	{
		grad += fe_V.shape_grad_component(i,q_point,component)*
			q[local_dof_index[i]];
	}

	return grad;

}


template <int dim>
Vector<double> elliptical<dim>::value_calc(const cell_iterator &cell,FEValues<dim> fe_V, Vector<double>q,uint q_point, uint dof)
{
	Vector<double>		grad;
	grad.reinit(dof);
	
	std::vector<types::global_dof_index> local_dof_index(dof_handler.n_dofs());
	cell->get_dof_indices(local_dof_index);
	grad=0;
	for(uint i = 0; i<dof;++i)
	{
		grad += fe_V.shape_value(i,q_point)*
			q[local_dof_index[i]];
	}

	return grad;

}

template <int dim>
void elliptical<dim>::run()
{
	make_grid();
	setup_system();
	assemble_system();
}

template <int dim>
void elliptical<dim>::output_data()
{
	DataOut<dim>	data_out;
	data_out.attach_dof_handler(dof_handler);

	std::ofstream output("results");
	data_out.write_vtk(output);
}

}//end of namespace



#endif
