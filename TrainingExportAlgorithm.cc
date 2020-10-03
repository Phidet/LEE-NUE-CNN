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
#include "larpandoracontent/LArHelpers/LArMCParticleHelper.h"
#include "Objects/MCParticle.h"
#include <fstream>
#include <array>
#include <limits>
#include <stdlib.h>
#include <time.h> 


#ifdef MONITORING
#include "PandoraMonitoringApi.h"
#endif


using namespace pandora;


namespace lar_content
{
	StatusCode TrainingExportAlgorithm::Run()
	{	
		const CaloHitList *pCaloHitListU(nullptr);
		const CaloHitList *pCaloHitListV(nullptr);
		const CaloHitList *pCaloHitListW(nullptr);		
		PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::GetList(*this, m_clusterListNames[0], pCaloHitListU));
		PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::GetList(*this, m_clusterListNames[1], pCaloHitListV));
		PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::GetList(*this, m_clusterListNames[2], pCaloHitListW));
		CaloHitVector caloHitVectorU(pCaloHitListU->begin(), pCaloHitListU->end());
		CaloHitVector caloHitVectorV(pCaloHitListV->begin(), pCaloHitListV->end());
		CaloHitVector caloHitVectorW(pCaloHitListW->begin(), pCaloHitListW->end());

		bool foundSuitableShower(false);

		const MCParticleList *pMCParticleList(nullptr);
	    CartesianVector vert = CartesianVector(0,0,0);
	    PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PandoraContentApi::GetCurrentList(*this, pMCParticleList));
	    int showers = 0;
	    int tracks = 0;
	    for (const MCParticle *const pMCParticle : *pMCParticleList)
	    {
	    	int mcPdg = pMCParticle->GetParticleId();

	    	if(pMCParticle->IsPfoTarget())
	    	{
	    		if(mcPdg==11 || mcPdg==-11 || mcPdg==22) showers++;
	    		else if(mcPdg==2212) tracks++;

		    	if(mcPdg!=2212 && mcPdg!=22 && mcPdg!=11 && mcPdg!=-11) return STATUS_CODE_SUCCESS;
		    	if(showers>1 || tracks>1) return STATUS_CODE_SUCCESS;
	    	}


	    	const int nuance = LArMCParticleHelper::GetNuanceCode(pMCParticle);
	        if (LArMCParticleHelper::IsNeutrino(pMCParticle) && (nuance==1001 || nuance==1000))
	        {
	        	vert = pMCParticle->GetVertex();
	        	const float innerRadius = pMCParticle->GetInnerRadius();
	        	const float outerRadius = pMCParticle->GetOuterRadius();
	        	const CartesianVector momentum =  pMCParticle->GetMomentum();
	        	//vert = CartesianVector(endpoint.GetX(), endpoint.GetY(), endpoint.GetZ());
	        	const CartesianVector mU = LArGeometryHelper::ProjectPosition(this->GetPandora(), momentum, TPC_VIEW_U); // Project 3D vertex onto 2D view
	        	const CartesianVector mV = LArGeometryHelper::ProjectPosition(this->GetPandora(), momentum, TPC_VIEW_V); // Project 3D vertex onto 2D view
	        	const CartesianVector mW = LArGeometryHelper::ProjectPosition(this->GetPandora(), momentum, TPC_VIEW_W); // Project 3D vertex onto 2D view

	        	std::cout<<"-|-|-|-|-|MC vertex: "<<vert.GetX()<<" "<<vert.GetY()<<" "<<vert.GetZ() <<" Momentum: "<<momentum.GetX()<<" "<<momentum.GetY()<<" "<<momentum.GetZ()<<" innerRadius: "<<innerRadius<<" outerRadius: "<<outerRadius<<std::endl;
	        	std::cout<<" mU: "<<mU.GetX()<<" "<<mU.GetY()<<" "<<mU.GetZ()<<std::endl;
	        	std::cout<<" mV: "<<mV.GetX()<<" "<<mV.GetY()<<" "<<mV.GetZ()<<std::endl;
	        	std::cout<<" mW: "<<mW.GetX()<<" "<<mW.GetY()<<" "<<mW.GetZ()<<std::endl;
	        	foundSuitableShower=true;
	        }
	    }



		if(foundSuitableShower)
		{
			float minX(0);
			float minZ_U(0), minZ_V(0), minZ_W(0);

			///////////////////////////////////////////////////////////////////////////////////////
			/// Find common minX
			const CartesianVector vertU = LArGeometryHelper::ProjectPosition(this->GetPandora(), vert, TPC_VIEW_U); // Project 3D vertex onto 2D U view
			const CartesianVector vertV = LArGeometryHelper::ProjectPosition(this->GetPandora(), vert, TPC_VIEW_V); // Project 3D vertex onto 2D V view
			const CartesianVector vertW = LArGeometryHelper::ProjectPosition(this->GetPandora(), vert, TPC_VIEW_W); // Project 3D vertex onto 2D W view

			float lowestX(std::numeric_limits<float>::max()-1.f);
			float highestX(-std::numeric_limits<float>::max()+1.f);
			float lowestZ_U(std::numeric_limits<float>::max()-1.f);
			float highestZ_U(-std::numeric_limits<float>::max()-1.f);
			float lowestZ_V(std::numeric_limits<float>::max()+1.f);
			float highestZ_V(-std::numeric_limits<float>::max()-1.f);
			float lowestZ_W(std::numeric_limits<float>::max()+1.f);
			float highestZ_W(-std::numeric_limits<float>::max()-1.f);

			int valuableMCHits(0);

			for (const CaloHit *const pCaloHit : caloHitVectorU)
			{
				const float x = pCaloHit->GetPositionVector().GetX();
				const float z = pCaloHit->GetPositionVector().GetZ();
				const MCParticleWeightMap  &mcParticleWeightMap(pCaloHit->GetMCParticleWeightMap());
				for (const MCParticleWeightMap::value_type &mapEntry : mcParticleWeightMap)
				{
					const int particleID = mapEntry.first->GetParticleId();	
					switch(particleID)
					{
						case 22: case 11: case -11: case 2212:
							valuableMCHits++;
							if(x<lowestX) lowestX=x;
							if(x>highestX) highestX=x;
							if(z<lowestZ_U) lowestZ_U=z;
							if(z>highestZ_U) highestZ_U=z;
							break;
					}
				}
			}

			for (const CaloHit *const pCaloHit : caloHitVectorV)
			{
				const float x = pCaloHit->GetPositionVector().GetX();
				const float z = pCaloHit->GetPositionVector().GetZ();
				const MCParticleWeightMap  &mcParticleWeightMap(pCaloHit->GetMCParticleWeightMap());
				for (const MCParticleWeightMap::value_type &mapEntry : mcParticleWeightMap)
				{
					const int particleID = mapEntry.first->GetParticleId();	
					switch(particleID)
					{
						case 22: case 11: case -11: case 2212:
							valuableMCHits++;
							if(x<lowestX) lowestX=x;
							if(x>highestX) highestX=x;
							if(z<lowestZ_V) lowestZ_V=z;
							if(z>highestZ_V) highestZ_V=z;
							break;
					}
				}
			}

			for (const CaloHit *const pCaloHit : caloHitVectorW)
			{
				const float x = pCaloHit->GetPositionVector().GetX();
				const float z = pCaloHit->GetPositionVector().GetZ();
				const MCParticleWeightMap  &mcParticleWeightMap(pCaloHit->GetMCParticleWeightMap());
				for (const MCParticleWeightMap::value_type &mapEntry : mcParticleWeightMap)
				{
					const int particleID = mapEntry.first->GetParticleId();	
					switch(particleID)
					{
						case 22: case 11: case -11: case 2212:
							valuableMCHits++;
							if(x<lowestX) lowestX=x;
							if(x>highestX) highestX=x;
							if(z<lowestZ_W) lowestZ_W=z;
							if(z>highestZ_W) highestZ_W=z;
							break;
					}
				}
			}

			if(valuableMCHits<15) return STATUS_CODE_SUCCESS;

			// initialize random seed
  			srand (time(NULL));
  			for(int i=0; i<1; i++)
  			{

  				const int diffX = (int)((highestX-lowestX)/0.3f);
  				const int diffZ_U = (int)((highestZ_U-lowestZ_U)/0.3f);
  				const int diffZ_V = (int)((highestZ_V-lowestZ_V)/0.3f);
  				const int diffZ_W = (int)((highestZ_W-lowestZ_W)/0.3f);



  				int randX(IMSIZE/4);
  				int randZ_U(IMSIZE/4);
  				int randZ_V(IMSIZE/4);
  				int randZ_W(IMSIZE/4);


  				if(diffX<IMSIZE) randX = IMSIZE-diffX;
				if(abs(vertU.GetX()-lowestX)<abs(highestX-vertU.GetX()))
					minX = lowestX - (rand()%randX)*0.3f;
				else
					minX = highestX -IMSIZE*0.3f + (rand()%randX)*0.3f;

  				if(diffZ_U<IMSIZE) randZ_U = IMSIZE-diffZ_U;
				if(abs(vertU.GetZ()-lowestZ_U)<abs(highestZ_U-vertU.GetZ()))
					minZ_U = std::min(vertU.GetZ()-5*0.4f, lowestZ_U - (rand()%randZ_U)*0.3f);
				else
					minZ_U = highestZ_U -IMSIZE*0.3f + (rand()%randZ_U)*0.3f;

  				if(diffZ_V<IMSIZE) randZ_V = IMSIZE-diffZ_V;
				if(abs(vertV.GetZ()-lowestZ_V)<abs(highestZ_V-vertV.GetZ()))
					minZ_V = lowestZ_V - (rand()%randZ_V)*0.3f;
				else
					minZ_V = highestZ_V -IMSIZE*0.3f + (rand()%randZ_V)*0.3f;

  				if(diffZ_W<IMSIZE) randZ_W = IMSIZE-diffZ_W;
				if(abs(vertW.GetZ()-lowestZ_W)<abs(highestZ_W-vertW.GetZ()))
					minZ_W = lowestZ_W - (rand()%randZ_W)*0.3f;
				else
					minZ_W = highestZ_W -IMSIZE*0.3f + (rand()%randZ_W)*0.3f;

			
				if(minX>vertU.GetX()-10*0.3f) minX=vertU.GetX()-10*0.3f;
				if(minX<vertU.GetX()-(IMSIZE-10)*0.3f) minX=vertU.GetX()-(IMSIZE-10)*0.3f;
				if(minZ_U>vertU.GetZ()-10*0.3f) minZ_U=vertU.GetZ()-10*0.3f;
				if(minZ_U<vertU.GetZ()-(IMSIZE-10)*0.3f) minZ_U=vertU.GetZ()-(IMSIZE-10)*0.3f;
				if(minZ_V>vertV.GetZ()-10*0.3f) minZ_V=vertV.GetZ()-10*0.3f;
				if(minZ_V<vertV.GetZ()-(IMSIZE-10)*0.3f) minZ_V=vertV.GetZ()-(IMSIZE-10)*0.3f;
				if(minZ_W>vertW.GetZ()-10*0.3f) minZ_W=vertW.GetZ()-10*0.3f;
				if(minZ_W<vertW.GetZ()-(IMSIZE-10)*0.3f) minZ_W=vertW.GetZ()-(IMSIZE-10)*0.3f;
			

				PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, WriteDetectorGaps(minZ_U, minZ_V, minZ_W));

				PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PopulateImage(caloHitVectorU, minX, minZ_U));
				PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PopulateImage(caloHitVectorV, minX, minZ_V));
				PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PopulateImage(caloHitVectorW, minX, minZ_W));
				//std::cout<<"++++++++++++ ++++++++++++ ++++++++++++ Point 9"<<std::endl;
				// PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PopulateRecoImage(*pPfoList, pCaloHitListU, vertU, TPC_VIEW_U));
				// PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PopulateRecoImage(*pPfoList, pCaloHitListV, vertV, TPC_VIEW_V));
				// PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, PopulateRecoImage(*pPfoList, pCaloHitListW, vertW, TPC_VIEW_W));
			}
		}
		return STATUS_CODE_SUCCESS;
	}

	StatusCode TrainingExportAlgorithm::WriteDetectorGaps(const float minZ_U, const float minZ_V, const float minZ_W)
	{
		std::array<float, 3*IMSIZE> gaps_UVW = {0};
		float minZ(0.f);
		for (const DetectorGap *const pDetectorGap : this->GetPandora().GetGeometry()->GetDetectorGapList())
		{
			const LineGap *const pLineGap = dynamic_cast<const LineGap*>(pDetectorGap);
        	if (!pLineGap) throw StatusCodeException(STATUS_CODE_INVALID_PARAMETER);

			const int gapType = static_cast<int>(pLineGap->GetLineGapType());
			switch(gapType)
			{
			case TPC_WIRE_GAP_VIEW_U: //gapType==0
				minZ = minZ_U;
				break;
			case TPC_WIRE_GAP_VIEW_V: //gapType==1
				minZ = minZ_V;
				break;
			case TPC_WIRE_GAP_VIEW_W: //gapType==2
				minZ = minZ_W;
				break;
			default:
				std::cout<<"Undeclared linegap type in TrainingExportAlgorithm::WriteDetectorGaps." <<std::endl;
				return STATUS_CODE_FAILURE;
			}

			const int gapStart = std::max(0,(int)((pLineGap->GetLineStartZ()-minZ)/0.3f));
			const int gapEnd = std::min(IMSIZE-1,(int)((pLineGap->GetLineEndZ()-minZ)/0.3f));
			for(int i=gapStart; i<=gapEnd; i++)
			{
				gaps_UVW[IMSIZE*gapType+i] = 1.f;
			}
		}

		std::ofstream file("OutTest2/viewUVW.bin", std::ios::out | std::ios::binary | std::ios::app); 
		if(!file)
		{
			std::cout<<"Problem opening/creating binary file in TrainingExportAlgorithm::PopulateImage."<<std::endl;
			return STATUS_CODE_FAILURE;
		}

		const float startMarker = std::numeric_limits<float>::max();
		file.write((char*)&startMarker, sizeof(startMarker));
		file.write((char*)&gaps_UVW, sizeof(gaps_UVW));
		file.close();

		return STATUS_CODE_SUCCESS;
	}


	void TrainingExportAlgorithm::fillMinimizationArray(std::array<float, SEG> &hitDensity, const PfoList *const pPfoList, const CaloHitList *const pCaloHitList, const CartesianVector v, const float startD1, const float startD2, const bool directionX, const HitType TPC_VIEW)
	{
		float weight, d1, d2;

		for (const ParticleFlowObject *const pPfo : *pPfoList) // Finds and adds shower to pfoListCrop
		{
			if (LArPfoHelper::IsShower(pPfo)) // && LArPfoHelper::IsNeutrinoFinalState(pPfo)
			{
				if(LArPfoHelper::IsNeutrinoFinalState(pPfo)) weight = 1.f;
				else weight = 0.f;
			}
			else
			{
				if(LArPfoHelper::IsNeutrinoFinalState(pPfo)) weight = 2.f;
				else weight = 0.f;
			}

			try
			{
				CartesianVector v2 =  LArPfoHelper::GetVertex(pPfo)->GetPosition();
				v2 = LArGeometryHelper::ProjectPosition(this->GetPandora(), v2, TPC_VIEW); // Project 3D vertex onto 2D view
				const float xDiff = v.GetX()-v2.GetX();
				const float zDiff = v.GetZ()-v2.GetZ();
				const float squaredDist = xDiff*xDiff+zDiff*zDiff;
				if(squaredDist>2000) weight *= 1.f;

			} 
				catch(StatusCodeException &statusCodeException)
			{
				std::cout<<"TrainingExportAlgorithm::fillMinimizationArray: No Pfo Vertex Found"<<std::endl;
			}


			PfoList pfoListTemp;
			pfoListTemp.push_back(pPfo);
			CaloHitList caloHitList;
			LArPfoHelper::GetCaloHits(pfoListTemp, TPC_VIEW, caloHitList);
			for (const CaloHit *const pCaloHit : caloHitList)
			{
				if(directionX){
					d1 = pCaloHit->GetPositionVector().GetX();
					d2 = pCaloHit->GetPositionVector().GetZ();
				} else {
					d1 = pCaloHit->GetPositionVector().GetZ();
					d2 = pCaloHit->GetPositionVector().GetX();
				}
				const int pixel = static_cast<int>(((d1-startD1)/0.3f + IMSIZE)/(2.0*IMSIZE)*SEG);
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
				const int pixel = static_cast<int>(((d1-startD1)/0.3f + IMSIZE)/(2.0*IMSIZE)*SEG);
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
				total += hitDensity[j]-hitDensity[i];
				if(total>0.f)
				{
					best = i;
					total = 0.f;
				}
			}

		return ((2.0*best)/SEG-1) * IMSIZE * 0.3f + startPoint;
	}


	StatusCode TrainingExportAlgorithm::PopulateImage(const CaloHitVector &caloHitVector, const float minX, const float minZ)
	{
		std::ofstream file("OutTest2/viewUVW.bin", std::ios::out | std::ios::binary | std::ios::app); 
		if(!file)
		{
			std::cout<<"Problem opening/creating binary file in TrainingExportAlgorithm::PopulateImage."<<std::endl;
			return STATUS_CODE_FAILURE;
		}

		const float hitNumber = -std::numeric_limits<float>::max();//caloHitVector.size();
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
			//std::cout<<"--------------------- New Hit"<<std::endl;
			for (const MCParticleWeightMap::value_type &mapEntry : mcParticleWeightMap)
			{
				const int particleID = mapEntry.first->GetParticleId();
				//std::cout<<"--------------------- particleID: "<<particleID<<" mapEntry.second"<<mapEntry.second<<std::endl;
				switch(particleID)
				{
					case 22: case 11: case -11:
						pixel[3] += mapEntry.second;
						break;
					case 2212:
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


	StatusCode TrainingExportAlgorithm::PopulateRecoImage(const PfoList &pfoList, const CaloHitList *pCaloHitList, const CartesianVector v, const HitType TPC_VIEW)
	{
		std::ofstream file("OutTest2/PandoraRecoUVW.bin", std::ios::out | std::ios::binary | std::ios::app); 
		if(!file)
		{
			std::cout<<"Problem opening/creating binary file in TrainingExportAlgorithm::PopulateImage."<<std::endl;
			return STATUS_CODE_FAILURE;
		}

		const float hitNumber = -std::numeric_limits<float>::max();
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
			LArPfoHelper::GetCaloHits(pfoListTemp, TPC_VIEW, caloHitList);
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
	// PANDORA_RETURN_RESULT_IF(STATUS_CODE_SUCCESS, !=, XmlHelper::ReadValue(xmlHandle, "PfoListNames", m_pfoListNames));

	PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadVectorOfValues(xmlHandle,
        "PfoListNames", m_pfoListNames));
	PANDORA_RETURN_RESULT_IF_AND_IF(STATUS_CODE_SUCCESS, STATUS_CODE_NOT_FOUND, !=, XmlHelper::ReadVectorOfValues(xmlHandle,
        "CaloHitListNames", m_clusterListNames));

    if (m_clusterListNames.empty())
    {
        std::cout << "TrainingExportAlgorithm::ReadSettings - Must provide names of cluster lists for use in U-Net." << std::endl;
        return STATUS_CODE_INVALID_PARAMETER;
    }

		return STATUS_CODE_SUCCESS;
	}
} // namespace lar_content
