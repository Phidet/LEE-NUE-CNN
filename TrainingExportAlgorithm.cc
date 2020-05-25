/** 
*  @file   larpandoracontent/LArWorkshop/TrainingExportAlgorithm.cc 
* 
*  @brief  Implementation of the TrainingExport algorithm class. 
* 
*  $Log: $ */
#include "Pandora/AlgorithmHeaders.h"
#include "larpandoracontent/MyArea/TrainingExportAlgorithm.h"
#include "PandoraMonitoringApi.h"
#include "larpandoracontent/LArHelpers/LArPfoHelper.h"
#include "larpandoracontent/LArHelpers/LArClusterHelper.h"
#include "larpandoracontent/LArHelpers/LArInteractionTypeHelper.h"
#include "larpandoracontent/LArHelpers/LArMonitoringHelper.h"
#include "larpandoracontent/LArHelpers/LArGeometryHelper.h"
#include "Objects/MCParticle.h"
#include <fstream>
#include <array>
#include <limits>

#ifdef MONITORING
#include "PandoraMonitoringApi.h"
#endif

//#include "larpandora/LArPandoraInterface/LArPandoraGeometry.h"

using namespace pandora;


namespace lar_content
{
	StatusCode TrainingExportAlgorithm::Run()
	{	
		//const CaloHitList *pCaloHitListU(nullptr);
		const CaloHitList *pCaloHitListV(nullptr);
		//const CaloHitList *pCaloHitListW(nullptr);
		//PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::GetList(*this, m_caloHitListNameU, pCaloHitListU));
		PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::GetList(*this, m_caloHitListNameV, pCaloHitListV));
		//PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::GetList(*this, m_caloHitListNameW, pCaloHitListW));

		//CaloHitVector caloHitVectorU(pCaloHitListU->begin(), pCaloHitListU->end());
		CaloHitVector caloHitVectorV(pCaloHitListV->begin(), pCaloHitListV->end());
		//CaloHitVector caloHitVectorW(pCaloHitListW->begin(), pCaloHitListW->end());

		const PfoList *pPfoList(nullptr);
		PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::GetCurrentList(*this, pPfoList));

		float minX(0);
		float minZ(0);
		float vertexX(std::numeric_limits<float>::max());
		float vertexZ(std::numeric_limits<float>::max());

		for (const ParticleFlowObject *const pPfo : *pPfoList) // Finds and adds shower to pfoListCrop
		{
			if (LArPfoHelper::IsShower(pPfo) && LArPfoHelper::IsNeutrinoFinalState(pPfo)) // && LArPfoHelper::IsNeutrinoFinalState(pPfo)
			{	
				PfoList pfoListTemp; // This block skips reconstructed showers with fewer than 15 hits in the V plane
				pfoListTemp.push_back(pPfo);
				CaloHitList caloHitList;
				LArPfoHelper::GetCaloHits(pfoListTemp, TPC_VIEW_V, caloHitList);
				if(caloHitList.size()<15) continue;

				CartesianVector v =  LArPfoHelper::GetVertex(pPfo)->GetPosition();
				v = LArGeometryHelper::ProjectPosition(this->GetPandora(), v, TPC_VIEW_V); // Project 3D vertex onto 2D U view
				
				const float xDiff = v.GetX()-vertexX;
				const float zDiff = v.GetZ()-vertexZ;
				const float minSquaredSeperation = (IMSIZE*0.3*0.66)*(IMSIZE*0.3*0.66);

				if(xDiff*xDiff+zDiff*zDiff>minSquaredSeperation)
				{
					vertexX = v.GetX();
					vertexZ = v.GetZ();
				    std::array<float, SEG>  hitXDensity= {0}; // Always combining 8 wires
				    fillMinimizationArray(hitXDensity, pPfoList, pCaloHitListV, v, vertexX, vertexZ-IMSIZE/3*0.3, true);
				    minX = findMin(hitXDensity, vertexX);

					std::array<float, SEG>  hitZDensity= {0}; // Always combining 8 wires
					fillMinimizationArray(hitZDensity, pPfoList, pCaloHitListV, v, vertexZ, minX, false);
					minZ = findMin(hitZDensity, vertexZ);

					// std::array<float, SEG>  hitX2Density= {0}; // Always combining 8 wires
					// fillMinimizationArray(hitX2Density, pPfoList, pCaloHitListU, v, vertexX, minZ, true);
					// minX = findMin(hitX2Density, vertexX);

					PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PopulateImage(caloHitVectorV, minX, minZ));
					PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PopulateRecoImage(*pPfoList, v));
				}
			}
		}
		return STATUS_CODE_SUCCESS;
	}

	void TrainingExportAlgorithm::fillMinimizationArray(std::array<float, SEG> &hitDensity, const PfoList *const pPfoList, const CaloHitList *const pCaloHitList, const CartesianVector v, const float startD1, const float startD2, const bool directionX)
	{
		float weight, d1, d2;

		for (const ParticleFlowObject *const pPfo : *pPfoList) // Finds and adds shower to pfoListCrop
		{
			if (LArPfoHelper::IsShower(pPfo)) // && LArPfoHelper::IsNeutrinoFinalState(pPfo)
			{
				if(LArPfoHelper::IsNeutrinoFinalState(pPfo)) weight = 1.f;
				else weight = 0.0f;
			}
			else
			{
				if(LArPfoHelper::IsNeutrinoFinalState(pPfo)) weight = 2.f;
				else weight = 0.0f;
			}

			try
			{
				CartesianVector v2 =  LArPfoHelper::GetVertex(pPfo)->GetPosition();
				v2 = LArGeometryHelper::ProjectPosition(this->GetPandora(), v2, TPC_VIEW_V); // Project 3D vertex onto 2D U view
				const float xDiff = v.GetX()-v2.GetX();
				const float zDiff = v.GetZ()-v2.GetZ();
				const float squaredDist = xDiff*xDiff+zDiff*zDiff;
				if(squaredDist>2000) weight *= 1.0;//6000.0/(squaredDist+4000.0);
				std::cout<<"Weight: "<<weight<<"  sqdst: "<<squaredDist<<std::endl;
			} 
				catch(StatusCodeException &statusCodeException)
			{
				std::cout<<"TrainingExportAlgorithm::fillMinimizationArray: No Pfo Vertex Found"<<std::endl;
			}


			PfoList pfoListTemp;
			pfoListTemp.push_back(pPfo);
			CaloHitList caloHitList;
			LArPfoHelper::GetCaloHits(pfoListTemp, TPC_VIEW_V, caloHitList);
			for (const CaloHit *const pCaloHit : caloHitList)
			{
				if(directionX){
					d1 = pCaloHit->GetPositionVector().GetX();
					d2 = pCaloHit->GetPositionVector().GetZ();
				} else {
					d1 = pCaloHit->GetPositionVector().GetZ();
					d2 = pCaloHit->GetPositionVector().GetX();
				}
				const int pixel = static_cast<int>(((d1-startD1)/0.3 + IMSIZE)/(2.0*IMSIZE)*SEG);
				if(pixel>=0 && pixel<SEG && (d2-startD2)/0.3<IMSIZE && (d2-startD2)>=0)
					hitDensity[pixel]+=weight;
			}
		}
		weight = 0.2f;
		for (const CaloHit *const pCaloHit : *pCaloHitList)
		{
			if(!PandoraContentApi::IsAvailable(*this, pCaloHit))
			{	
				if(directionX){
					d1 = pCaloHit->GetPositionVector().GetX();
					d2 = pCaloHit->GetPositionVector().GetZ();
				} else {
					d1 = pCaloHit->GetPositionVector().GetZ();
					d2 = pCaloHit->GetPositionVector().GetX();
				}
				const int pixel = static_cast<int>(((d1-startD1)/0.3 + IMSIZE)/(2.0*IMSIZE)*SEG);
				if(pixel>=0 && pixel<SEG && (d2-startD2)/0.3<IMSIZE && (d2-startD2)>=0)
					hitDensity[pixel]+=weight;
			}
		}
	}

	float TrainingExportAlgorithm::findMin(const std::array<float, SEG> hitDensity, const float startPoint) const
	{
		float total(0.f);
		int best = 0;
		for(int i=0; i<SEG/2; i++)
			{
				const int j = SEG/2+i;
				std::cout<< "X - i: "<< i <<" j: "<<j<<" hD[j]: "<<hitDensity[j]<<" hD[i]: "<<hitDensity[i]<<" total: "<<total<<" best: "<<best<<std::endl;
				total += hitDensity[j]-hitDensity[i];
				if(total>0.f)
				{
					best = i;
					total = 0.f;
				}
			}

		return ((2.0*best)/SEG-1) * IMSIZE * 0.3 + startPoint;


		// int left(0);
		// int middle(SEG/4);
		// int right(SEG/2);
		// int loopCounter(1);
		// int leftTotal, rightTotal, middleTotal;
		// do{
		// 	leftTotal = 0;
		// 	middleTotal = 0;
		// 	rightTotal = 0;
		// 	for(int i=0; i<SEG/2; i++)
		// 	{
		// 		leftTotal += hitDensity[left+i];
		// 		middleTotal += hitDensity[middle+i];
		// 		rightTotal += hitDensity[right+i];
		// 	}

		// 	if(middleTotal<leftTotal || middleTotal<rightTotal)
		// 	{
		// 		middle = leftTotal>rightTotal ? left:right;
		// 	}
		// 	left = middle - (SEG/8)/loopCounter;
		// 	right = middle + (SEG/8)/loopCounter;

		// 	left = left>0 ? left: 0;
		// 	right = right<SEG/2 ? right: SEG/2;

		// 	loopCounter *=2;
		// } while(loopCounter<(SEG/4));
		// const float minValue = ((2.0*middle)/SEG-1) * IMSIZE * 0.3 + startPoint;
		// return minValue;
	}


	StatusCode TrainingExportAlgorithm::PopulateImage(const CaloHitVector &caloHitVector, const float minX, const float minZ)
	{
		std::ofstream file("OutTest/viewV.bin", std::ios::out | std::ios::binary | std::ios::app); 
		if(!file)
		{
			std::cout<<"Problem opening/creating binary file in TrainingExportAlgorithm::PopulateImage."<<std::endl;
			return STATUS_CODE_FAILURE;
		}

		const float hitNumber = -1.22f;//caloHitVector.size();
		file.write((char*)&hitNumber, sizeof(hitNumber));
		file.write((char*)&minX, sizeof(minX));
		file.write((char*)&minZ, sizeof(minZ));

		for (const CaloHit *const pCaloHit : caloHitVector)
		{
			const float x = pCaloHit->GetPositionVector().GetX();
			const float z = pCaloHit->GetPositionVector().GetZ();
			if((x-minX)/0.3>=IMSIZE || (z-minZ)/0.3>=IMSIZE || (x-minX)<0 || (z-minZ)<0) continue; // Skipps hits that are not in the crop area
			
			std::array<float, 6> pixel = {0};
			pixel[0] = x;
			pixel[1] = z; 
			pixel[2] = pCaloHit->GetHadronicEnergy(); // Populates input image

			const MCParticleWeightMap  &mcParticleWeightMap(pCaloHit->GetMCParticleWeightMap());
			// Populates prediction image
			for (const MCParticleWeightMap::value_type &mapEntry : mcParticleWeightMap)
			{
				const int particleID = mapEntry.first->GetParticleId();
				switch(particleID)
				{
					case 22: case 11: case -11:
					pixel[3] += mapEntry.second;
					break;
					case 2212: case 211: case -211:
					pixel[4] += mapEntry.second;
					break;
				}
			}
			pixel[5] = 1.0f - pixel[3] - pixel[4];
			file.write((char*)&pixel, sizeof(pixel));
		}
		file.close();
		return STATUS_CODE_SUCCESS;
	}	


	StatusCode TrainingExportAlgorithm::PopulateRecoImage(const PfoList &pfoList, const CartesianVector v)
	{
		std::ofstream file("OutTest/PandoraRecoV.bin", std::ios::out | std::ios::binary | std::ios::app); 
		if(!file)
		{
			std::cout<<"Problem opening/creating binary file in TrainingExportAlgorithm::PopulateImage."<<std::endl;
			return STATUS_CODE_FAILURE;
		}

		const float hitNumber = -1.22f;
		file.write((char*)&hitNumber, sizeof(hitNumber));
		const float vertexX = v.GetX();
		const float vertexZ = v.GetZ();

		file.write((char*)&vertexX, sizeof(vertexX));
		file.write((char*)&vertexZ, sizeof(vertexZ));

		for (const ParticleFlowObject *const pPfo: pfoList)
		{
			PfoList pfoListTemp;
			pfoListTemp.push_back(pPfo);
			CaloHitList caloHitList;
			LArPfoHelper::GetCaloHits(pfoListTemp, TPC_VIEW_V, caloHitList);
			for (const CaloHit *const pCaloHit : caloHitList)
			{
				std::array<float, 4> pixel = {0};
	    		const float x = pCaloHit->GetPositionVector().GetX(); //(pCaloHit->GetPositionVector().GetX()-minX)/widthX; // Pixel number in X direction
	    		const float z = pCaloHit->GetPositionVector().GetZ(); //(pCaloHit->GetPositionVector().GetZ()-minZ)/widthZ; // Pixel number in Z direction
	    		file.write((char*)&x, sizeof(x));
	    		file.write((char*)&z, sizeof(z));

	    		if(LArPfoHelper::IsShower(pPfo))
	    		{
	    			if(LArPfoHelper::IsNeutrinoFinalState(pPfo))
	    			{
	    				pixel[0] = 1.0f;
	    			}
	    			else
	    			{
	    				pixel[1] = 1.0f;	
	    			}
	    		}
	    		else if(LArPfoHelper::IsTrack(pPfo))
	    		{
	    			if(LArPfoHelper::IsNeutrinoFinalState(pPfo))
	    			{
	    				pixel[2] = 1.0f;
	    			}
	    			else
	    			{
	    				pixel[3] = 1.0f;	
	    			}
	    		}
	    		file.write((char*)&pixel, sizeof(pixel));
	    	}
	    }
	    file.close();
	    return STATUS_CODE_SUCCESS;
	}	


//------------------------------------------------------------------------------------------------------------------------------------------
	StatusCode TrainingExportAlgorithm::ReadSettings(const TiXmlHandle xmlHandle)
	{
	// Read settings from xml file here
	//PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, XmlHelper::ReadValue(xmlHandle, "PfoListName", m_pfoListName));
		PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, XmlHelper::ReadValue(xmlHandle, "CaloHitListNameU", m_caloHitListNameU));
		PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, XmlHelper::ReadValue(xmlHandle, "CaloHitListNameV", m_caloHitListNameV));
		PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, XmlHelper::ReadValue(xmlHandle, "CaloHitListNameW", m_caloHitListNameW));

		return STATUS_CODE_SUCCESS;
	}
} // namespace lar_content
