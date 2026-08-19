import { ComponentFixture, TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting } from '@angular/common/http/testing';
import { ActivatedRoute } from '@angular/router';
import { provideRouter } from '@angular/router';
import { PersonInvestigation } from './person-investigation';

describe('PersonInvestigation', () => {
  let component: PersonInvestigation;
  let fixture: ComponentFixture<PersonInvestigation>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [PersonInvestigation],
      providers: [
        provideHttpClient(),
        provideHttpClientTesting(),
        provideRouter([]),
        { provide: ActivatedRoute, useValue: { snapshot: { paramMap: new Map([['id', '1']]) } } },
      ],
    }).compileComponents();

    fixture = TestBed.createComponent(PersonInvestigation);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
